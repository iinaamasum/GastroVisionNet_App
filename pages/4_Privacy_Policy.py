import streamlit as st
from utils.ui import inject_base_styles, sidebar_brand, footer


def render():
    st.set_page_config(
        page_title="GastroVisionNet - Privacy", page_icon=None, layout="wide"
    )
    # sidebar_brand()
    inject_base_styles()

    st.title("GastroVisionNet")
    st.subheader("Privacy Policy")

    st.markdown(
        "<div class='card'><h3>Intended Use</h3><p>GastroVisionNet is designed for research and educational purposes to assist in image-based endoscopic analysis. It is not a substitute for professional medical judgment yet. It is still under development and is not yet ready for clinical use. A rigorous validation process is ongoing to ensure the accuracy and reliability of the model.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card'><h3>Approved Environments</h3><p>Can be used in research labs, academic settings, and controlled clinical evaluations with appropriate oversight by a medical professional.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card'><h3>Not For</h3><p>Primary diagnosis, autonomous decision-making, or unsupervised clinical deployment. It is not yet ready for clinical use.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card'><h3>Data Policy</h3><p>Uploaded images are processed in-memory for inference and not persisted by the application. No PHI should be uploaded. Users remain responsible for data compliance.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card'><h3>Rights</h3><p>All rights reserved by Qatar University. The team reserves the right to change features, models, and policies without prior notice.</p></div>",
        unsafe_allow_html=True,
    )

    footer()


render()

