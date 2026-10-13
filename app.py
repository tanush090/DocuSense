
import io
from pathlib import Path

import fitz
import joblib
import pytesseract
import streamlit as st
from PIL import Image

# ===================== CONFIG =====================

st.set_page_config(
    page_title="DocuSense | AI Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "document_classifier.pkl"

# ===================== DESIGN SYSTEM =====================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #090b13;
    --panel: #121523;
    --border: #282d43;
    --muted: #a1a8bf;
    --purple: #a99bff;
    --mint: #62e7d0;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #211b49 0%, transparent 38%),
        radial-gradient(ellipse at 95% 12%, #10313b 0%, transparent 30%),
        var(--bg);
    color: #f5f6ff;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.7px;
}

[data-testid="stSidebar"] {
    background: #0e101b;
    border-right: 1px solid var(--border);
}

.block-container {
    max-width: 1250px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 22px 0 30px;
}

.eyebrow {
    color: var(--purple);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(42px, 6vw, 72px);
    font-weight: 700;
    letter-spacing: -3px;
    line-height: 1.05;
    margin: 18px 0;
}

.gradient {
    background: linear-gradient(90deg, #c2b5ff, #62e7d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: var(--muted);
    font-size: 16px;
    line-height: 1.8;
    max-width: 720px;
}

.panel {
    background: rgba(18, 21, 35, 0.88);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 23px;
    margin-bottom: 15px;
}

.panel-label {
    color: var(--purple);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.muted {
    color: var(--muted);
    line-height: 1.7;
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
}

[data-testid="stFileUploader"] {
    background: rgba(18, 21, 35, 0.8);
    border: 1px dashed #625b8d;
    border-radius: 16px;
    padding: 12px;
}

.stButton > button {
    background: linear-gradient(100deg, #b5a5ff, #62e7d0);
    color: #0b0d15;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 46px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}

.stDownloadButton > button {
    background: #191c2d;
    color: #f5f6ff;
    border: 1px solid #373b57;
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

.footer {
    border-top: 1px solid var(--border);
    color: #8289a3;
    text-align: center;
    font-size: 12px;
    padding-top: 24px;
    margin-top: 38px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ===================== MODEL =====================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


# ===================== EXTRACTION =====================

def extract_pdf_text(file_bytes):
    """Extract selectable PDF text and use OCR for scanned pages."""
    pages_text = []

    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        for page in document:
            text = page.get_text("text").strip()

            if len(text) < 30:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                text = pytesseract.image_to_string(image)

            pages_text.append(text)

    return "\n".join(pages_text).strip()


def extract_image_text(file_bytes):
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return pytesseract.image_to_string(image).strip()


def display_label(label):
    label = str(label).strip()
    normalized = label.lower()

    for supported in ("bill", "invoice", "receipt"):
        if supported in normalized:
            return supported.title()

    return label


def predict_document(text):
    prediction = model.predict([text])[0]
    confidence_scores = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        confidence_scores = {
            display_label(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

    return display_label(prediction), confidence_scores


# ===================== SIDEBAR =====================

with st.sidebar:
    st.markdown("## ◈ DocuSense")
    st.caption("AI DOCUMENT INTELLIGENCE")
    st.divider()

    st.markdown("### Capabilities")
    st.markdown(
        "- 📄 PDF text extraction\n"
        "- 👁️ OCR for scanned files\n"
        "- 🧾 Bill classification\n"
        "- 📑 Invoice classification\n"
        "- 🛍️ Receipt classification"
    )

    st.divider()
    st.markdown("### Workflow")
    st.markdown(
        "01 — Upload a document\n\n"
        "02 — Extract its content\n\n"
        "03 — Run the ML model\n\n"
        "04 — Review the result"
    )

    st.divider()
    st.caption("Created by Tanush Kumar")


# ===================== HERO =====================

st.markdown("""
<div class="hero">
    <div class="eyebrow">INTELLIGENT DOCUMENT ANALYSIS</div>
    <div class="hero-title">
        Documents in.<br>
        <span class="gradient">Clarity out.</span>
    </div>
    <div class="subtitle">
        Transform everyday financial documents into meaningful insights.
        Upload a bill, invoice or receipt and let machine learning
        analyze its content in seconds.
    </div>
</div>
""", unsafe_allow_html=True)


# ===================== FEATURE CARDS =====================

c1, c2, c3 = st.columns(3)

features = [
    ("01 / INPUT", "Upload", "PDF documents and images"),
    ("02 / ANALYSIS", "Understand", "Text extraction and OCR"),
    ("03 / OUTPUT", "Classify", "Bill, invoice or receipt"),
]

for col, (number, title, description) in zip((c1, c2, c3), features):
    with col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">{number}</div>
                <h3>{title}</h3>
                <div class="muted">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===================== UPLOAD =====================

st.markdown("## Analyze your document")
st.markdown(
    '<p class="muted">Upload a document to begin the classification process.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, PNG, JPG and JPEG.",
)

if model is None:
    st.error(
        "Model not found. Add document_classifier.pkl to the same "
        "repository folder as app.py."
    )
    st.stop()


# ===================== RESULTS =====================

if uploaded_file is not None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">SELECTED FILE</div>
            <h3>{uploaded_file.name}</h3>
            <div class="muted">{uploaded_file.size / 1024:.1f} KB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✦ Analyze document", use_container_width=True):
        try:
            with st.spinner("Extracting text and analyzing your document..."):
                file_bytes = uploaded_file.getvalue()

                if uploaded_file.name.lower().endswith(".pdf"):
                    extracted_text = extract_pdf_text(file_bytes)
                else:
                    extracted_text = extract_image_text(file_bytes)

                if not extracted_text.strip():
                    st.error(
                        "No readable text was found. Try a clearer file."
                    )
                    st.stop()

                predicted_label, scores = predict_document(extracted_text)

            st.success("Analysis complete")

            left, right = st.columns([1.2, 1])

            with left:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-label">PREDICTED DOCUMENT TYPE</div>
                        <h2>◈ {predicted_label}</h2>
                        <div class="muted">
                            Classification generated by your trained model.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                confidence = scores.get(predicted_label)
                if confidence is not None:
                    st.metric("Model confidence", f"{confidence * 100:.2f}%")
                else:
                    st.info("Confidence scores are unavailable.")

            if scores:
                st.markdown("### Confidence breakdown")

                for label, score in sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                ):
                    st.write(f"**{label}** · {score * 100:.2f}%")
                    st.progress(max(0.0, min(1.0, score)))

            with st.expander("View extracted text"):
                st.text_area(
                    "Extracted content",
                    extracted_text,
                    height=250,
                )

            st.download_button(
                "↓ Download extracted text",
                data=extracted_text,
                file_name=f"{Path(uploaded_file.name).stem}_extracted.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception:
            st.error(
                "Document analysis failed. Check the uploaded file and "
                "try again. Scanned documents may need clearer images."
            )
            st.exception(Exception("See the application logs for details."))


# ===================== FOOTER =====================

st.markdown("""
<div class="footer">
    DOCUSENSE · AI-POWERED DOCUMENT CLASSIFICATION<br><br>
    Designed and developed by Tanush Kumar
</div>
""", unsafe_allow_html=True)
