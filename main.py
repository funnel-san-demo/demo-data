import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import pickle
import zipfile
import base64

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import get_column_plot

from faker import Faker
from faker.providers import BaseProvider

fake = Faker()

import random
import nltk

nltk.download("words")
from nltk.corpus import words
import re

class CustomProvider(BaseProvider):
    def __init__(self, provider):
        self.words = words.words()

    def fake_dimensions(self, prefix):
        return f"{prefix} - {random.sample(self.words, 1)[0].capitalize()}"

    def generate_random_sequence(self, length):
        first_digit = str(random.randint(1, 9))
        rest_of_digits = "".join(str(random.randint(0, 9)) for _ in range(length - 1))
        return first_digit + rest_of_digits


fake.add_provider(CustomProvider)


def create_metadata_obj(dataset):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)
    return metadata


def create_converter(file):
    header_row = file.readline().strip().split(",")
    id_columns = [col.replace('"', "") for col in header_row if col.replace('"', "").endswith(" ID")]
    converters = {col: str for col in id_columns}
    return converters


def get_dimensions(metadata):
    dimensions = []
    for dimension, data_type in metadata.to_dict()["columns"].items():
        if data_type["sdtype"] == "categorical":
            dimensions.append(dimension)
    return dimensions


def generate_constraints(constraint_class, _dimensions):
    return {
        "constraint_class": constraint_class,
        "constraint_parameters": {"column_names": _dimensions},
    }


def generate_fake_campaign_values(product_category, product, color, market, audience):
    return f"{random.choice(product_category)} | {random.choice(product)} | {random.choice(color)} | {random.choice(market)} | {random.choice(audience)}"


def save_dataframe_to_csv_zip(dataframe, zip_file, file_name):
    # Save the dataframe to a CSV file in memory
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Add the CSV file to the zip file
    zip_file.writestr(file_name, csv_buffer.getvalue())


def get_binary_file_downloader_html(bin_data, file_label='File', btn_label='Download'):
    """Generates a link to download a binary file."""
    bin_str = bin_data.getvalue()
    bin_str = base64.b64encode(bin_str).decode()
    href = f'data:application/zip;base64,{bin_str}'
    return f'<a href="{href}" download="{file_label}.zip">{btn_label}</a>'

df = None
dim_cols = None
product_category, product, color, market, audience = None, None, None, None, None




st.subheader("1. Preprocess data")
with st.expander("Select your dataset that you want to anonymize"):
    uploaded_file_preprocess = st.file_uploader("Choose a file:", key="file_uploader_1")

    if uploaded_file_preprocess is not None:
        df = pd.read_csv(
            uploaded_file_preprocess,
            converters=create_converter(
                StringIO(uploaded_file_preprocess.getvalue().decode("utf-8"))
            ),
        )
        st.write("First 10 rows of your selected dataset")
        st.write(df.head(10))

        metadata = create_metadata_obj(df)
        dimensions = get_dimensions(metadata)
     
        dim_cols = st.multiselect('Choose dimensions you want to scramble', dimensions, key=1)

        if any(s in dim_cols for s in ['Campaign', 'Campaign Name']):
            if scramble_campaign:=st.checkbox("Customize output for Campaign/Campaign Name"):
                product_category = [item.strip() for item in st.text_input('Enter a comma separated list of product categories').split(',')]
                product = [item.strip() for item in st.text_input('Enter a comma separated list of products').split(',')]
                color = [item.strip() for item in st.text_input('Enter a comma separated list of colors').split(',')]
                market = [item.strip() for item in st.text_input('Enter a comma separated list of markets').split(',')]
                audience = [item.strip() for item in st.text_input('Enter a comma separated list of audiences').split(',')]

        if st.button("Scramble dimensions"):
            with st.spinner("Please wait while we scramble your dimensions..."):
                for column in dim_cols:
                    if column.endswith(" ID"):
                        fake_values = [fake.generate_random_sequence(16) for _ in range(len(df[column].unique()))]
                    elif column in ['Campaign', 'Campaign Name'] and scramble_campaign:
                        fake_values = [
                            generate_fake_campaign_values(product_category, product, color, market, audience)
                            for _ in range(len(df[column].unique()))
                        ]
                    else:
                        fake_values = [fake.fake_dimensions(column) for _ in range(len(df[column].unique()))]

                    df[column] = df[column].replace(dict(zip(df[column].unique(), fake_values)), regex=False)
            st.success("Success!")
            st.download_button("Click to download", df.to_csv(index=False).encode("utf-8"), "anonymized_data.csv", "text/csv")


st.subheader("2. Create a trained model for generating synthetic data")
with st.expander("Select the dataset you want to use to train the synthetic data model"):
    uploaded_file = st.file_uploader("Choose a file:")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, converters=create_converter(StringIO(uploaded_file.getvalue().decode("utf-8"))))
        st.write("First 10 rows of your selected dataset")
        st.write(df.head(10))

        metadata = create_metadata_obj(df)

        for dimension, data_type in metadata.to_dict()["columns"].items():
            if data_type["sdtype"] == "unknown":
                metadata.update_column(column_name=dimension, sdtype="categorical")
        dimensions = get_dimensions(metadata)
        constraints = []
        constraints.append(generate_constraints("Unique", dimensions + ["Date"]))
        id_columns = [column for column in df.columns if column.endswith(" ID")]

        for id in id_columns:
            if id.replace(" ID", "") in dimensions:
                constraints.append(generate_constraints("FixedCombinations", [id, id.replace(" ID", "")]))
            elif id.replace(" ID", " Name") in dimensions:
                constraints.append(generate_constraints("FixedCombinations", [id, id.replace(" ID", " Name")]))

        if st.button("Click here to start the training process"):
            with st.spinner("Please wait while your synthesizer trains. This might take time"):
                synthesizer = CTGANSynthesizer(metadata)
                synthesizer.add_constraints(constraints=constraints)
                synthesizer.fit(df)

            st.success("Synthesizer was trained succesfully!")

            def pickle_model(model):
                f = BytesIO()
                pickle.dump(model, f)
                return f

            data = pickle_model(synthesizer)
            st.download_button("Download your model", data=data, file_name="synthetic_data_model.pkl")


st.subheader("3. Generate synthetic data from a trained model")
with st.expander("Generate synthetic data"):
    file_path = st.file_uploader("Upload Pickle File", type=["pkl"])
    if file_path is not None:
        with BytesIO(file_path.getvalue()) as f:
            model = pickle.load(f)

        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("Number of samples to generate", min_value=0, value=1000)
        with col2:
            n_files = st.number_input("Number of files to generate", min_value=1, value=1)

        if st.button("Generate samples"):
            # with st.spinner("Generating samples... This might take time."):
            #     synth_data = model.sample(num_rows=n_samples)
            # st.success(f"Data was succesfully generated!")
            # st.download_button("Download", synth_data.to_csv(index=False).encode("utf-8"), "synthetic_data.csv", "text/csv")

            with st.spinner("Generating samples... This might take time."):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i in range(n_files):
                        synth_data = model.sample(num_rows=n_samples)
                        save_dataframe_to_csv_zip(synth_data, zip_file, f'file_{i+1}.csv')
            zip_buffer.seek(0)
            st.success(f"Data was succesfully generated!")
            st.markdown(get_binary_file_downloader_html(zip_buffer, 'Zip File', 'download'), unsafe_allow_html=True)
        


            



