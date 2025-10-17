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
    for i, path in enumerate(items):
        with cols[i % 3]:
            st.image(path, caption=os.path.basename(path), width="stretch")

    footer()


render()

