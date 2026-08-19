import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import catdogcnn


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CatDog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODEL
# ============================================================

device = torch.device("cpu")

model = catdogcnn(3)

model.load_state_dict(
    torch.load(
        "cat_dog_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def classify_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probability = torch.sigmoid(output).item()

    if probability >= 0.5:

        label = "DOG"
        confidence = probability

    else:

        label = "CAT"
        confidence = 1 - probability

    return label, confidence * 100


# ============================================================
# PROFESSIONAL COMPACT CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       HIDE STREAMLIT DEFAULT TOP HEADER
       ======================================================== */

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    header[data-testid="stHeader"] button {
        display: none;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(79, 70, 229, 0.15),
                transparent 28%
            ),

            radial-gradient(
                circle at 90% 15%,
                rgba(14, 165, 233, 0.12),
                transparent 28%
            ),

            #0f172a;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {

        max-width: 1180px;

        padding-top: 1.8rem;

        padding-bottom: 0.45rem;

        padding-left: 1.5rem;

        padding-right: 1.5rem;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {

        color: #f8fafc !important;

        font-size: 30px !important;

        margin-top: 0 !important;

        margin-bottom: 0.1rem !important;
    }


    h2 {

        color: #e2e8f0 !important;

        font-size: 19px !important;

        margin-top: 0.2rem !important;

        margin-bottom: 0.35rem !important;
    }


    h3 {

        color: #cbd5e1 !important;

        font-size: 16px !important;
    }


    p {

        color: #94a3b8;

        font-size: 13px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {

        background: rgba(30, 41, 59, 0.72);

        border: 1px solid rgba(148, 163, 184, 0.13);

        border-radius: 11px;

        padding: 8px 12px;

        min-height: 65px;
    }


    div[data-testid="stMetricLabel"] {

        color: #94a3b8 !important;

        font-size: 11px !important;
    }


    div[data-testid="stMetricValue"] {

        color: #f8fafc !important;

        font-size: 21px !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    div[data-testid="stFileUploader"] {

        background: rgba(30, 41, 59, 0.55);

        border-radius: 11px;

        padding: 5px;

        border: 1px solid rgba(99, 102, 241, 0.28);
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {

        width: 100%;

        border-radius: 9px;

        border: none;

        background: linear-gradient(
            90deg,
            #6366f1,
            #0ea5e9
        );

        color: white;

        font-weight: 700;

        padding: 7px 12px;

        font-size: 13px;

        min-height: 38px;
    }


    .stButton > button:hover {

        background: linear-gradient(
            90deg,
            #4f46e5,
            #0284c7
        );

        color: white;
    }


    /* ========================================================
       IMAGE
       ======================================================== */

    img {

        border-radius: 10px;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {

        margin: 0.45rem 0 !important;

        border-color: rgba(
            148,
            163,
            184,
            0.12
        );
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {

        padding: 8px 12px;

        border-radius: 9px;

        font-size: 12px;
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    div[data-testid="stProgressBar"] {

        margin-top: 3px;

        margin-bottom: 5px;
    }


    /* ========================================================
       CAPTION
       ======================================================== */

    .stCaption {

        font-size: 11px !important;
    }


    /* ========================================================
       COLUMN SPACING
       ======================================================== */

    div[data-testid="column"] {

        padding-left: 0.3rem;

        padding-right: 0.3rem;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🐾 CatDog Classifier"
)

st.caption(
    "Computer Vision • CNN Image Classification"
)


# ============================================================
# TOP STATS
# ============================================================

stat1, stat2, stat3, stat4 = st.columns(4)


with stat1:

    st.metric(
        "Test Accuracy",
        "83.88%"
    )


with stat2:

    st.metric(
        "Model",
        "CNN"
    )


with stat3:

    st.metric(
        "Classes",
        "2"
    )


with stat4:

    st.metric(
        "Input Size",
        "256 × 256"
    )


st.divider()


# ============================================================
# MAIN SECTION
# ============================================================

left, middle, right = st.columns(
    [1.05, 1.05, 0.9],
    gap="small"
)


# ============================================================
# LEFT — UPLOAD
# ============================================================

with left:

    st.subheader(
        "📤 Upload"
    )

    st.caption(
        "JPG • JPEG • PNG"
    )

    uploaded_file = st.file_uploader(
        "Choose image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    else:

        st.info(
            "👆 Upload an image to begin."
        )


# ============================================================
# MIDDLE — PREDICTION
# ============================================================

with middle:

    st.subheader(
        "🧠 Prediction"
    )

    st.caption(
        "CNN model analysis"
    )


    if uploaded_file is None:

        st.info(
            "Prediction will appear here."
        )


    else:

        if st.button(
            "🚀 Classify Image"
        ):

            with st.spinner(
                "Analyzing..."
            ):

                label, confidence = classify_image(
                    image
                )


            if label == "CAT":

                st.success(
                    "🐱 CAT"
                )

            else:

                st.success(
                    "🐶 DOG"
                )


            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            st.progress(
                min(
                    int(confidence),
                    100
                )
            )


            st.caption(
                "✓ Classification completed"
            )


# ============================================================
# RIGHT — MODEL INFORMATION
# ============================================================

with right:

    st.subheader(
        "📊 Model"
    )


    st.info(
        "CNN Architecture"
    )


    st.write(
        """
        **Input:** 256 × 256 RGB

        **Conv Layers:** 3

        **Channels:** 32 → 64 → 128

        **Output:** Cat / Dog
        """
    )


    st.success(
        "✓ Model loaded"
    )


# ============================================================
# PIPELINE
# ============================================================

st.divider()


st.subheader(
    "⚙️ Pipeline"
)


p1, p2, p3, p4, p5 = st.columns(5)


with p1:

    st.info(
        "📷 INPUT"
    )


with p2:

    st.info(
        "📐 RESIZE"
    )


with p3:

    st.info(
        "🔢 TENSOR"
    )


with p4:

    st.info(
        "🧠 CNN"
    )


with p5:

    st.info(
        "🐱🐶 RESULT"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "CatDog Classifier • PyTorch • Streamlit • Computer Vision"
)