 🎧 EmoSense: Emotion Detection from Speech

Welcome to EmoSense – an intelligent web app that detects human emotions from speech using machine learning, deployed on Streamlit Cloud, and powered by MFCC features + Random Forest classification, this project demonstrates the power of combining audio processing with deep learning.

---
🚀 Live Demo

👉 [Click here to try EmoSense](https://emosense-app-mokshi.streamlit.app)  
🎤 Upload any `.wav` or `.mp3` audio sample and get the predicted emotion instantly!

---

📌 Project Highlights

- 🎙️ Accepts audio input in `.wav` or `.mp3` formats
- 🔊 Extracts 40 MFCC features using `librosa`
- 🌲 Uses a Random Forest Classifier for emotion prediction
- 🧠 Trained on the RAVDESS dataset (augmented for better accuracy)
- 🎯 Achieved **96.41% test accuracy**
- 📊 Includes waveform and mel-spectrogram visualizations
- 💾 Model stored using `pickle` and loaded dynamically in Streamlit
- 🔁 MP3 support with `pydub` and `ffmpeg` added via `packages.txt`

---

 🧠 Detected Emotions

EmoSense can recognize **8 distinct emotions**:

- 😐 Neutral
- 😌 Calm
- 😄 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

---

 🛠️ Tech Stack

| Tool | Role |
|------|------|
| **Python 3.8** | Core language |
| **Librosa** | Audio feature extraction |
| **Scikit-learn** | Model training (Random Forest) |
| **Streamlit** | Interactive web UI |
| **pydub** | MP3 support |
| **Git LFS** | Handles large model files |
| **ffmpeg** | Backend audio processing |

---


📁 Repository Structure

