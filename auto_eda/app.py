import streamlit as st
import pandas as pd
from auto_eda import AutoEDA

st.title("📊 Automated EDA Tool")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    eda = AutoEDA(df)

    st.subheader("Overview")
    st.write(df.shape)
    st.write(df.dtypes)
    st.write(df.isnull().sum())
    st.write(df.describe(include='all'))

    if st.checkbox("Show Univariate Analysis"):
        eda.univariate_analysis()

    if st.checkbox("Show Bivariate Analysis"):
        target = st.selectbox("Select Target Column", df.columns)
        eda.bivariate_analysis(target=target)

    if st.checkbox("Show Correlation Heatmap"):
        eda.correlation_heatmap()

    if st.checkbox("Detect Outliers"):
        method = st.selectbox("Method", ["zscore", "iqr"])
        eda.detect_outliers(method=method)

    if st.checkbox("Analyze Time-Series"):
        datetime_col = st.selectbox("Datetime Column", df.columns)
        value_col = st.selectbox("Value Column", df.select_dtypes(include='number').columns)
        freq = st.text_input("Frequency", value="D")
        eda.analyze_time_series(datetime_col=datetime_col, value_col=value_col, freq=freq)