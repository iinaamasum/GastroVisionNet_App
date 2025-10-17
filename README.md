# GastroVisionNet

Modern Streamlit app for endoscopic image analysis with clear navigation, card-based content and unified branding.

## Run locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Launch the app from the workspace root:
    ```bash
    streamlit run GastroVisionNet/Home.py
    ```

## Navigation

-   Evaluate Model (`pages/1_Evaluate_Model.py`)
-   Sample Images (`pages/2_Sample_Images.py`)
-   How to Use (`pages/3_How_to_Use.py`)
-   Privacy Policy (`pages/4_Privacy_Policy.py`)

The sidebar is branded as “GastroVisionNet” and the first page is selected by default. Icons have been removed for a professional, consistent look.

## Assets

-   Models and checkpoints should be placed as required by `utils/models.py` (e.g., `MNv2_SA_8_cls/Best_States/best_model_fold_3.pth`).
-   Sample images: `GastroVisionNet/assets/samples/`
-   Tutorial video: `GastroVisionNet/assets/video/howto.mp4`

## UI Notes

-   Card components are used across pages, and the hero width matches the card width for visual consistency.
-   A common footer appears on every page.
