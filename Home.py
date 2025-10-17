import streamlit as st
from utils.ui import inject_base_styles, footer


st.set_page_config(page_title="GastroVisionNet", page_icon=None, layout="wide")
inject_base_styles()


def main():
    hero = """
    <div class='hero-wrap'>
      <div class='hero-card'>
        <div style='font-size:40px;font-weight:800;margin-bottom:6px;'>GastroVisionNet</div>
        <div style='font-size:18px;opacity:0.95;'>Precise gastrointestinal endoscopy image analysis powered by deep learning</div>
        <div style='margin-top:8px;font-size:14px;'>Developed by Qatar University Research Team, led by Dr. Amith Khandakar</div>
      </div>
    </div>
    """
    st.markdown(hero, unsafe_allow_html=True)

    st.markdown(
        "<div class='card'><h3>Key Features</h3><ul><li><strong> Evaluate Model:</strong> Multi-scheme classification (8-class, 6-class, polyps vs. non-polyps, normal vs. abnormal classification)</li><li><strong>Sample Images:</strong> View sample endoscopic images used for testing and evaluate the model</li><li><strong>How to Use:</strong> Learn how to use the system effectively</li><li><strong>Privacy Policy:</strong> Understand how your data is handled</li></ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card'><h3>System Requirements</h3><ul><li>Modern browser (Chrome/Firefox/Safari/Edge)</li><li>Internet for model loading</li><li>Image formats: PNG, JPG, JPEG</li></ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='notice'>Use the sidebar to navigate: Evaluate Model, Sample Images, How to Use, Privacy Policy.</div>",
        unsafe_allow_html=True,
    )

    footer()


if __name__ == "__main__":
    main()

