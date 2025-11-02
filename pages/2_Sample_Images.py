import os
from glob import glob
import streamlit as st
from utils.ui import inject_base_styles, sidebar_brand, footer


def _load_samples(root: str):
    gallery_dir = os.path.join(root, "assets", "samples")
    return sorted(glob(os.path.join(gallery_dir, "*.png")))


def render():
    st.set_page_config(
        page_title="GastroVisionNet - Samples", page_icon=None, layout="wide"
    )
    inject_base_styles()
    # sidebar_brand()

    st.title("GastroVisionNet")
    st.subheader("Sample Images")
    st.write(
        "Explore representative images across classes. Drop more images into the gallery folder to extend."
    )

    items = _load_samples(os.getcwd())
    if not items:
        st.info(
            "No sample images found yet. Drop images into `GastroVisionNet/assets/samples/` to display here."
        )
        footer()
        return

    cols = st.columns(3)
    info_dic = {
        0: "Polyps | Abnormal",
        1: "Z-Line | Normal | Non-Polyps",
        2: "Pylorus | Normal | Non-Polyps",
        3: "Cecum | Normal | Non-Polyps",
        4: "Esophagitis | Abnormal | Non-Polyps",
        5: "Ulcerative-Colitis | Abnormal | Non-Polyps",
        6: "Dyed-Lifted-Polyps",
        7: "Dyed-Resection-Margins",
    }
    for i, path in enumerate(items):
        with cols[i % 3]:
            st.image(path, caption=os.path.basename(path), width="stretch")
            st.info(f"{info_dic[i % 8]}", width="stretch")

    footer()


render()
