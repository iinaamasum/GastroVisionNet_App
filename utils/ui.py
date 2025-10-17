import streamlit as st


def inject_base_styles():
    st.markdown(
        """
        <style>
        /* Hide default sidebar section header */
        section[data-testid="stSidebar"] h2 { display: none; }

        /* Card layout */
        .card { 
          background: #083344; 
          border: 1px solid #0e7490; 
          border-radius: 12px; 
          padding: 18px; 
          margin-bottom: 16px; 
          max-width: 860px; 
          margin-left: auto; 
          margin-right: auto;
        }
        .card h3 { margin: 0 0 8px 0; }

        .notice { 
          background:#052e3b; 
          border:1px solid #145369; 
          border-radius:12px; 
          padding:14px; 
          max-width: 860px; 
          margin: 0 auto 16px auto;
        }

        /* Hero shares the same width as card */
        .hero-card {
          display:inline-block; 
          padding:24px 32px; 
          border-radius:16px; 
          background:linear-gradient(90deg,#0ea5e9,#6366f1); 
          color:white; 
          max-width: 860px; 
          width: 100%;
        }
        .hero-wrap { text-align:center; padding:32px 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand():
    st.sidebar.markdown(
        """
        <div style='padding:8px 6px 16px 6px;'>
          <div style='font-weight:800;font-size:18px;letter-spacing:0.2px;'>GastroVisionNet</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        "<div style='text-align:center;color:#6b7280;padding:24px 0;'>"
        "© 2025 Qatar University Research Team<br/>"
        "GastroVisionNet v1.0 | Advanced Endoscopic Image Intelligence<br/>"
        "Contact: <a href='mailto:masum.cse19@gmail.com'>masum.cse19@gmail.com</a>"
        "</div>",
        unsafe_allow_html=True,
    )

