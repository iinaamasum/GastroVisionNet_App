import os
import streamlit as st
from utils.ui import inject_base_styles, sidebar_brand, footer


def render():
    st.set_page_config(
        page_title="GastroVisionNet - How to Use", page_icon=None, layout="wide"
    )
    inject_base_styles()
    # sidebar_brand()

    st.title("GastroVisionNet")
    st.subheader("How to Use")
    st.write(
        "This short video demonstrates the workflow. Please play the video to see the workflow."
    )

    video_path = "assets/video/Video_Presentation.mp4"
    if os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                st.video(f.read(), format="video/mp4")
        except Exception:
            st.markdown(
                f"""
<video width="100%" controls autoplay muted playsinline>
  <source src="file://{video_path}" type="video/mp4">
  Your browser does not support the video tag.
</video>
""",
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "Place a tutorial video at `GastroVisionNet/assets/video/howto.mp4` to enable autoplay demo."
        )

    footer()


render()
