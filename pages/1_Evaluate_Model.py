import streamlit as st

from utils.ui import inject_base_styles, sidebar_brand, footer
from utils.models import load_model, predict_image, MODEL_REGISTRY
from utils.gradcam_utils import GradCAM, overlay_heatmap
import torchvision.transforms as T
from PIL import Image
import numpy as np


def _to_bgr(np_rgb):
    return np_rgb[:, :, ::-1]


def _get_preprocess():
    return T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _ensure_model(scheme: str):
    if "model_state" not in st.session_state or st.session_state.model_state is None:
        with st.spinner("Loading model..."):
            st.session_state.model_state = load_model(scheme)
    else:
        _, _, spec = st.session_state.model_state
        if spec.title != scheme:
            with st.spinner("Switching model..."):
                st.session_state.model_state = load_model(scheme)


def render():
    st.set_page_config(
        page_title="GastroVisionNet - Evaluate", page_icon=None, layout="wide"
    )
    inject_base_styles()
    # sidebar_brand()

    st.title("GastroVisionNet")
    st.subheader("Evaluate Model")

    all_schemes = list(MODEL_REGISTRY.keys())
    filtered = [
        s
        for s in all_schemes
        if s not in ("Anatomical Classification", "Pathological Classification")
    ]
    scheme = st.selectbox("Select Classification Scheme", filtered, index=0)

    _ensure_model(scheme)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if st.button("Evaluate", disabled=uploaded is None):
        if uploaded is None:
            st.warning("Please upload an image first.")
            return
        image = Image.open(uploaded).convert("RGB")
        preprocess = _get_preprocess()
        tensor = preprocess(image)

        model, device, spec = st.session_state.model_state
        pred_idx, probs, class_name = predict_image(
            model, device, tensor, spec.task_type, spec.class_names
        )

        if spec.task_type == "binary":
            st.success(f"✅ Prediction: {class_name}")
            # st.write(f"Confidence: {float(probs):.3f}")

        else:
            st.success(f"✅ Prediction: {class_name}")
            # st.write(f"Confidence: {probs[pred_idx]:.3f}")

            import pandas as pd

            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame(
                {"Class": spec.class_names, "Probability": probs.numpy()}
            )
            st.bar_chart(prob_df.set_index("Class"))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width="stretch")
        with col2:
            with st.spinner("Generating Grad-CAM..."):
                input_batch = tensor.unsqueeze(0).to(device)
                target_layer = (
                    model.features[-1]
                    if hasattr(model, "features")
                    else (
                        model.enhanced_blocks[-1]
                        if hasattr(model, "enhanced_blocks")
                        else list(model.modules())[-1]
                    )
                )
                cam = GradCAM(model, target_layer)
                class_idx = pred_idx if spec.task_type == "multiclass" else 0
                heatmap = cam.generate(input_batch, class_idx)
                img_np = np.array(image)
                img_bgr = _to_bgr(img_np)
                overlay = overlay_heatmap(img_bgr, heatmap, alpha=0.45)
                overlay_rgb = overlay[:, :, ::-1]
                st.image(
                    overlay_rgb,
                    caption=f'Grad-CAM for {spec.title} where predicted class is "{class_name}"',
                    width="stretch",
                )
    st.markdown(
        """
        <div class='card'>
            <h3>Sorry for the Initial Loading Time</h3>
            <p>
                We are using Streamlit to host the initial version of this application. 
                It uses a remote setup, which requires downloading the entire GitHub repository. 
                As a result, the loading process may take some time. 
                Please be patient while the application loads. 
                It may take approximately 2–5 minutes, depending on your internet connection. 
            </p>
            <strong>***Once all the files are downloaded, the application will not require this loading time again.***</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer()


render()
