import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import tempfile
import os

# Attempt to import pydub (for MP3 conversion)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ==================== Custom CSS ====================
# We set a simple white background and darker text for high contrast.
custom_css = """
<style>
body {
    background-color: #ffffff; 
    color: #000000; 
    font-family: 'Segoe UI', sans-serif;
    margin: 0; 
    padding: 0;
}
h1, h2, h3 {
    color: #000000;
}
hr {
    border: 1px solid #38bdf8;
    margin: 1.5rem 0;
}
.stFileUploader {
    border: 2px dashed #38bdf8;
    border-radius: 10px;
    padding: 2rem;
    background-color: #f9f9f9;
}
.stButton>button {
    background-color: #38bdf8;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-size: 16px;
}
.container {
    width: 90%;
    max-width: 1200px;
    margin: auto;
}
footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #666;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==================== Big Heading First ====================
st.markdown("<h1 style='text-align:center;'>EmoSense: Emotion Detection from Speech</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==================== Hero Image (new 1.jpg) ====================
hero_image = "new 1.jpg"
if os.path.exists(hero_image):
    st.image(hero_image, use_container_width=True)
else:
    st.warning(f"Hero image '{hero_image}' not found!")

# ==================== Short Description ====================
st.markdown("""
<div style='text-align:center; margin-top:1rem;'>
    <p style='font-size:1.2rem; max-width:700px; margin:auto; line-height:1.6;'>
    Experience a premium audio emotion analysis service. 
    Upload your audio (WAV or MP3) to discover the emotion behind the voice.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==================== Container Start ====================
st.markdown("<div class='container'>", unsafe_allow_html=True)

# ==================== Load Model ====================
MODEL_PATH = "rf_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model file 'rf_model.pkl' not found. Please run the training script first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ==================== Emotion Mapping ====================
emotion_mapping = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}

# ==================== File Uploader + Clear ====================
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

uploaded_file_temp = st.file_uploader("Upload your audio file", type=["wav", "mp3"])
if uploaded_file_temp is not None:
    st.session_state["uploaded_file"] = uploaded_file_temp

# Clear button
if st.session_state["uploaded_file"] is not None:
    if st.button("Clear File"):
        st.session_state["uploaded_file"] = None
        st.experimental_rerun()

if st.session_state["uploaded_file"] is not None:
    # Remove mime_type for older Streamlit versions
    st.audio(st.session_state["uploaded_file"])

    # Handle MP3 → WAV conversion if needed
    file_ext = st.session_state["uploaded_file"].name.split(".")[-1].lower()
    if file_ext == "mp3":
        if not PYDUB_AVAILABLE:
            st.error("pydub not installed. Please install pydub to handle MP3 files.")
            st.stop()
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio = AudioSegment.from_file(st.session_state["uploaded_file"], format="mp3")
                audio.export(tmp.name, format="wav")
                audio_path = tmp.name
    else:
        audio_path = st.session_state["uploaded_file"]

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        st.success("Audio loaded successfully!")
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        st.stop()

    # Extract features & predict
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1).reshape(1, -1)
    prediction = model.predict(mfcc)
    predicted_class = prediction[0]
    emotion_label = emotion_mapping.get(predicted_class, "Unknown")

    st.subheader("Predicted Emotion:")
    st.markdown(f"<h2 style='text-align: center; color: #38bdf8;'>{emotion_label}</h2>", unsafe_allow_html=True)

    # Visualization
    tabs = st.tabs(["Waveform", "Spectrogram"])
    with tabs[0]:
        st.subheader("Audio Waveform")
        fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        # No facecolor override => default white background
        ax_wave.set_title("Waveform", color="#000000")
        ax_wave.set_xlabel("Time (s)", color="#000000")
        ax_wave.set_ylabel("Amplitude", color="#000000")
        # Spines default to black or grey
        st.pyplot(fig_wave)

    with tabs[1]:
        st.subheader("Mel-Spectrogram")
        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, ax=ax_spec, x_axis="time", y_axis="mel")
        ax_spec.set_title("Mel-Frequency Spectrogram", color="#000000")
        ax_spec.set_xlabel("Time (s)", color="#000000")
        ax_spec.set_ylabel("Mel Bin", color="#000000")
        st.pyplot(fig_spec)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==================== About the Detected Emotions Section ====================
st.markdown("<div class='container'>", unsafe_allow_html=True)
st.subheader("About the Detected Emotions")

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("""
    **What Emotions Does EmoSense Detect?**  
    <br>
    **Neutral** – A balanced state with no extreme emotion.  
    **Calm** – A peaceful, relaxed state.  
    **Happy** – High energy and positive vibes.  
    **Sad** – A low or down state.  
    **Angry** – High intensity and possibly aggressive tone.  
    **Fearful** – A state of worry or anxiety.  
    **Disgust** – A strong sense of aversion.  
    **Surprised** – An unexpected reaction.
    """, unsafe_allow_html=True)

with col2:
    # If your file is named "emotions.jpg"
    emotion_image = "emotions.jpg"
    if os.path.exists(emotion_image):
        st.image(emotion_image, use_container_width=True)
    else:
        st.warning(f"'{emotion_image}' not found!")

st.markdown("</div>", unsafe_allow_html=True)

# ==================== Footer ====================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<footer style='text-align: center; padding: 1rem;'>Built with ❤️ by <strong>Mokshitha</strong></footer>", unsafe_allow_html=True)
