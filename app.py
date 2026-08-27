"""
Brain Tumor Detection System
============================

A Streamlit application that classifies a brain MRI image into one of four
classes (glioma, meningioma, notumor, pituitary) using a fine-tuned VGG16
transfer-learning model that was trained separately.

The preprocessing here matches the training/evaluation pipeline exactly:
    load image -> resize to 224x224 -> to array -> add batch dim -> preprocess_input

No Rescaling(1./255) is applied, because VGG16's own preprocess_input was the
only normalisation used when the model was trained.

Run locally with:
    streamlit run app.py
"""

import os

# Quiet TensorFlow's startup logging. Must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

MODEL_FILENAME = "brain_tumor_vgg16.keras"
CLASS_NAMES_FILENAME = "class_names.json"

# The model and class-name files may sit next to app.py or inside a subfolder
# such as Models/. Checking a few sensible spots keeps the project portable no
# matter which directory Streamlit is launched from.
SEARCH_DIRS = [".", "Models", "models", "model", "saved_model", "artifacts"]

IMAGE_SIZE = (224, 224)

# Below this top-class probability the model is not making a decisive call,
# so the UI says so instead of presenting the result as settled.
DECISIVE_THRESHOLD = 60.0

# Reported on the held-out Testing set in the training notebook.
# Update this if the model is ever retrained.
TEST_ACCURACY_TEXT = "90.4%"

# Human-readable labels for the raw class names stored in class_names.json.
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

# One stable colour per class, so a colour always means the same thing.
CLASS_COLORS = {
    "glioma": "#D14343",
    "meningioma": "#7C5CD3",
    "notumor": "#0E8A6E",
    "pituitary": "#C77A0A",
}
FALLBACK_COLOR = "#55677A"

DISCLAIMER = (
    "This application is for educational/project purposes only and is not a "
    "medical diagnostic tool."
)


st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="\N{BRAIN}",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --ink: #0F1E2E;
  --slate: #55677A;
  --page: #F5F7FA;
  --card: #FFFFFF;
  --line: #DFE5EC;
  --accent: #0E7C86;
  --accent-soft: #E9F3F4;
}

.stApp { background: var(--page); }
header[data-testid="stHeader"], #MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem; }

.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--ink);
}

/* --- Small-caps utility label, used for every field name in the UI --- */
.bt-lab {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.66rem;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--slate);
  display: block;
  margin-bottom: 0.35rem;
}

/* --- Hero --- */
.bt-hero { margin-bottom: 1.6rem; }
.bt-hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(1.9rem, 4vw, 2.7rem);
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin: 0.35rem 0 0.7rem 0;
}
.bt-hero p {
  font-size: 1rem;
  line-height: 1.65;
  color: var(--slate);
  max-width: 62ch;
  margin: 0;
}
.bt-rule { height: 1px; background: var(--line); margin: 1.5rem 0; }

/* --- Model spec strip --- */
.bt-spec {
  display: flex;
  flex-wrap: wrap;
  gap: 2.4rem;
  padding: 0.9rem 0 0 0;
}
.bt-spec-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--ink);
}

/* --- Cards --- */
.bt-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1.4rem 1.5rem;
  margin-bottom: 1.1rem;
}
.bt-card h2 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.02rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 1.1rem 0;
}

/* --- Headline result --- */
.bt-result-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(2rem, 5vw, 2.9rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.05;
  margin: 0 0 1.3rem 0;
}
.bt-metrics { display: flex; gap: 2.8rem; flex-wrap: wrap; }
.bt-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.55rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}

/* --- Four-channel probability readout (the signature element) --- */
.bt-row {
  display: grid;
  grid-template-columns: 9px minmax(72px, 104px) 1fr 62px;
  align-items: center;
  gap: 0.85rem;
  padding: 0.5rem 0;
}
.bt-row + .bt-row { border-top: 1px solid #EEF1F5; }
.bt-dot { width: 9px; height: 9px; border-radius: 2px; display: block; }
.bt-name { font-size: 0.9rem; font-weight: 500; }
.bt-row--top .bt-name { font-weight: 600; }
.bt-track {
  height: 6px;
  background: #EDF0F4;
  border-radius: 3px;
  overflow: hidden;
  display: block;
}
.bt-fill { height: 6px; border-radius: 3px; display: block; }
.bt-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.83rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--slate);
}
.bt-row--top .bt-pct { color: var(--ink); font-weight: 700; }

/* --- Notices --- */
.bt-note {
  border-left: 2px solid var(--accent);
  background: var(--accent-soft);
  padding: 0.85rem 1.1rem;
  border-radius: 0 6px 6px 0;
  font-size: 0.88rem;
  line-height: 1.6;
  color: #12414A;
}
.bt-warn {
  border-left: 2px solid #C77A0A;
  background: #FDF6EA;
  padding: 0.85rem 1.1rem;
  border-radius: 0 6px 6px 0;
  font-size: 0.88rem;
  line-height: 1.6;
  color: #6B4405;
  margin-bottom: 1.1rem;
}
.bt-error {
  border-left: 2px solid #D14343;
  background: #FDEDED;
  padding: 1rem 1.2rem;
  border-radius: 0 6px 6px 0;
  font-size: 0.9rem;
  line-height: 1.65;
  color: #7A2020;
}
.bt-error code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.83rem;
  background: #FBDCDC;
  padding: 0.08rem 0.32rem;
  border-radius: 3px;
}

/* --- Empty state --- */
.bt-empty {
  background: var(--card);
  border: 1px dashed var(--line);
  border-radius: 10px;
  padding: 3.2rem 1.5rem;
  text-align: center;
}
.bt-empty-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.4rem;
}
.bt-empty p { font-size: 0.88rem; color: var(--slate); margin: 0; }

/* --- Footer --- */
.bt-foot {
  border-top: 1px solid var(--line);
  margin-top: 2.2rem;
  padding-top: 1.1rem;
  font-size: 0.8rem;
  color: var(--slate);
  line-height: 1.6;
}

/* --- Streamlit widget overrides --- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card);
  border: 1px dashed #C6D1DE;
  border-radius: 10px;
  padding: 1.4rem 1rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] button {
  background: var(--ink);
  color: #FFFFFF;
  border: none;
  border-radius: 6px;
  font-weight: 500;
}
[data-testid="stFileUploaderDropzone"] small { color: var(--slate); }
[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--line); }

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


def compact(html: str) -> str:
    """Collapse a readable HTML template onto one line.

    Streamlit's markdown renderer treats indented lines as code blocks, so the
    templates below are written for readability and flattened before rendering.
    """
    return " ".join(html.split())


# ---------------------------------------------------------------------------
# Loading the model and class names
# ---------------------------------------------------------------------------

def find_project_file(filename: str):
    """Return the path to a project file, or None if it cannot be found."""
    for folder in SEARCH_DIRS:
        candidate = APP_DIR / folder / filename
        if candidate.is_file():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model once per session. Returns (model, error_message)."""
    path = find_project_file(MODEL_FILENAME)
    if path is None:
        looked_in = ", ".join(SEARCH_DIRS)
        return None, (
            f"Could not find <code>{MODEL_FILENAME}</code>. Place it next to "
            f"<code>app.py</code> or in one of these subfolders: {looked_in}."
        )
    # A Git LFS pointer is a ~130-byte text file, not a model. Hosts that clone
    # without LFS support hand you one of these silently, and the resulting
    # error is unrecognisable. Check for it explicitly.
    size_bytes = path.stat().st_size
    if size_bytes < 1024 * 1024:
        return None, (
            f"<code>{path.name}</code> is only {size_bytes:,} bytes, which means "
            f"it is a <strong>Git LFS pointer file</strong> rather than the model "
            f"itself. The host cloned this repository without resolving LFS "
            f"objects.<br><br>Commit the model as a normal file instead — see "
            f"<code>slim_model.py</code>, which strips the unused optimiser state "
            f"and brings it under the 100 MB limit."
        )

    try:
        # compile=False skips rebuilding the training optimiser, which this app
        # never uses. Beyond loading faster, it avoids allocating Adam's ~79 MB
        # of state — worth having on a memory-capped host. Weights and
        # predictions are identical either way.
        return tf.keras.models.load_model(path, compile=False), None
    except Exception as exc:
        return None, (
            f"Found <code>{path.name}</code> but it could not be loaded. This is "
            f"usually a Keras version mismatch — the file was written by Keras "
            f"3.13.2, so <code>requirements.txt</code> needs "
            f"<code>keras&gt;=3.13.2</code>.<br><br>Details: <code>{exc}</code>"
        )


@st.cache_data(show_spinner=False)
def load_class_names():
    """Load class names in their saved order. Returns (class_names, error_message)."""
    path = find_project_file(CLASS_NAMES_FILENAME)
    if path is None:
        return None, (
            f"Could not find <code>{CLASS_NAMES_FILENAME}</code>. It must sit "
            f"alongside <code>{MODEL_FILENAME}</code>."
        )
    try:
        with open(path, "r", encoding="utf-8") as file:
            class_names = json.load(file)
        if not isinstance(class_names, list) or not class_names:
            return None, f"<code>{CLASS_NAMES_FILENAME}</code> is not a non-empty list."
        return class_names, None
    except Exception as exc:
        return None, f"Could not read <code>{CLASS_NAMES_FILENAME}</code>: <code>{exc}</code>"


def pretty_name(raw_name: str) -> str:
    """Map a raw class name to a display label."""
    return DISPLAY_NAMES.get(raw_name.strip().lower(), raw_name.replace("_", " ").title())


def class_color(raw_name: str) -> str:
    """Map a raw class name to its stable colour."""
    return CLASS_COLORS.get(raw_name.strip().lower(), FALLBACK_COLOR)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def preprocess_image(file_bytes: bytes):
    """Turn uploaded image bytes into a model-ready batch.

    Mirrors the training pipeline: resize to 224x224, convert to an array, add
    the batch dimension, then apply VGG16's preprocess_input. Nothing else.
    """
    image = tf.keras.utils.load_img(
        io.BytesIO(file_bytes),
        target_size=IMAGE_SIZE,
        color_mode="rgb",  # grayscale MRIs become 3-channel, as during training
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def predict(model, batch):
    """Run the model and return (probabilities, top_index, confidence_percent)."""
    probabilities = model.predict(batch, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities)) * 100
    return probabilities, top_index, confidence


def margin_over_next(probabilities) -> float:
    """Percentage-point gap between the top two classes.

    A large gap means the model clearly separated one class from the rest; a
    small gap means the top answer was nearly a tie.
    """
    if len(probabilities) < 2:
        return 100.0
    top_two = np.sort(probabilities)[::-1][:2]
    return float(top_two[0] - top_two[1]) * 100


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def show_image(image_source, caption: str) -> None:
    """Display an image full-width, across differing Streamlit versions."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True},
                   {"use_column_width": True}):
        try:
            st.image(image_source, caption=caption, **kwargs)
            return
        except TypeError:
            continue
    st.image(image_source, caption=caption)


def render_hero() -> None:
    st.markdown(compact(f"""
        <div class="bt-hero">
          <span class="bt-lab">Deep learning &middot; medical imaging</span>
          <h1>Brain Tumor Detection System</h1>
          <p>
            Upload a brain MRI slice and the model sorts it across four
            categories &mdash; glioma, meningioma, pituitary tumour, or no
            tumour. Predictions come from a VGG16 network pre-trained on
            ImageNet and fine-tuned on 5,600 labelled MRI scans. Every
            probability the model produces is shown, not just the winning one.
          </p>
        </div>
        <div class="bt-spec">
          <div><span class="bt-lab">Architecture</span>
               <span class="bt-spec-val">VGG16 fine-tuned</span></div>
          <div><span class="bt-lab">Input tensor</span>
               <span class="bt-spec-val">224 &times; 224 &times; 3</span></div>
          <div><span class="bt-lab">Classes</span>
               <span class="bt-spec-val">4</span></div>
          <div><span class="bt-lab">Test accuracy</span>
               <span class="bt-spec-val">{TEST_ACCURACY_TEXT}</span></div>
        </div>
        <div class="bt-rule"></div>
    """), unsafe_allow_html=True)


def render_result(class_names, probabilities, top_index, confidence) -> None:
    """Headline prediction, confidence, and the four-class breakdown."""
    top_name = pretty_name(class_names[top_index])
    top_color = class_color(class_names[top_index])
    margin = margin_over_next(probabilities)

    st.markdown(compact(f"""
        <div class="bt-card">
          <span class="bt-lab">Predicted class</span>
          <div class="bt-result-name" style="color:{top_color}">{top_name}</div>
          <div class="bt-metrics">
            <div><span class="bt-lab">Model confidence</span>
                 <span class="bt-num">{confidence:.2f}%</span></div>
            <div><span class="bt-lab">Margin over next</span>
                 <span class="bt-num">{margin:.2f} pts</span></div>
          </div>
        </div>
    """), unsafe_allow_html=True)

    # Fixed class order (as saved in class_names.json) so the readout stays
    # comparable from one image to the next.
    rows = []
    for index, raw_name in enumerate(class_names):
        percent = float(probabilities[index]) * 100
        color = class_color(raw_name)
        is_top = index == top_index
        rows.append(f"""
            <div class="bt-row {'bt-row--top' if is_top else ''}">
              <span class="bt-dot" style="background:{color}"></span>
              <span class="bt-name">{pretty_name(raw_name)}</span>
              <span class="bt-track">
                <span class="bt-fill" style="width:{max(percent, 0.7):.2f}%;
                      background:{color}"></span>
              </span>
              <span class="bt-pct">{percent:.2f}%</span>
            </div>
        """)

    st.markdown(compact(f"""
        <div class="bt-card">
          <h2>Probability across all four classes</h2>
          {''.join(rows)}
        </div>
    """), unsafe_allow_html=True)

    if confidence < DECISIVE_THRESHOLD:
        st.markdown(compact(f"""
            <div class="bt-warn">
              <strong>Not a decisive prediction.</strong> The top class reached
              only {confidence:.2f}%, so the model is splitting its estimate
              across categories rather than settling on one. Treat the
              breakdown above as the real output here.
            </div>
        """), unsafe_allow_html=True)

    st.markdown(compact(f"""
        <div class="bt-note">
          <strong>Model confidence is not medical certainty.</strong> It is the
          probability the network assigned to its top class. {DISCLAIMER}
        </div>
    """), unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(compact("""
        <div class="bt-empty">
          <div class="bt-empty-title">No MRI loaded yet</div>
          <p>Upload a JPG, JPEG, or PNG scan to see the prediction
             and the full probability breakdown.</p>
        </div>
    """), unsafe_allow_html=True)


def render_error(message: str) -> None:
    st.markdown(
        compact(f'<div class="bt-error">{message}</div>'),
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(compact(f"""
        <div class="bt-foot">
          <strong>{DISCLAIMER}</strong><br>
          It has not been clinically validated or reviewed by a medical
          professional, and must not be used to make decisions about anyone's
          health. Consult a qualified radiologist or physician for any question
          about a real scan.
        </div>
    """), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_hero()

    class_names, class_error = load_class_names()
    with st.spinner("Loading the model..."):
        model, model_error = load_model()

    # Without the model or the class names there is nothing to run.
    if class_error or model_error:
        render_error(class_error or model_error)
        render_footer()
        return

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<span class="bt-lab">Upload MRI scan</span>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Brain MRI image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="A single axial, coronal, or sagittal brain MRI slice.",
        )

    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

    # Preprocess first, so an unreadable file is caught before it is displayed.
    batch = None
    if file_bytes:
        try:
            batch = preprocess_image(file_bytes)
        except Exception:
            with right:
                render_error(
                    "That file could not be opened as an image. It may be "
                    "corrupted, or saved with a different format than its "
                    "extension suggests. Try re-exporting it as JPG or PNG."
                )

    with left:
        if batch is not None:
            show_image(file_bytes, f"Uploaded scan — {uploaded_file.name}")

    with right:
        if batch is None:
            if not file_bytes:
                render_empty_state()
        else:
            with st.spinner("Analysing scan..."):
                probabilities, top_index, confidence = predict(model, batch)
            render_result(class_names, probabilities, top_index, confidence)

    render_footer()


if __name__ == "__main__":
    main()
