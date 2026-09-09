import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk Model — Results", layout="wide")
st.title("Credit Risk: PD, LGD, EAD Model Results")


with open("artifacts/model_results.json") as f:
    results = json.load(f)

st.header("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("PD Model AUC", f"{results['pd_auc']:.3f}")
col2.metric("LGD Model RMSE", f"{results['lgd_rmse']:.3f}")
col3.metric("EAD Model RMSE", f"${results['ead_rmse']:,.0f}")

st.header("Business Impact: Provisioning")
col1, col2 = st.columns(2)
col1.metric("Naive Flat Reserve", f"${results['naive_reserve']:,.0f}")
col2.metric("Risk-Adjusted Reserve", f"${results['risk_adjusted_reserve']:,.0f}",
            delta=f"{results['efficiency_gain_pct']:.1f}%")

st.header("Portfolio Overview")
df = pd.read_parquet("artifacts/cleaned_loans_sample.parquet")  
st.bar_chart(df.groupby('grade')['default_flag'].mean())