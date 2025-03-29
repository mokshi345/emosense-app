import os
import librosa
import numpy as np
import pandas as pd

def pitch_shift_audio(y, sr, steps):
    # For librosa 0.11.0+, use keyword args:
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)

def time_stretch_audio(y, rate):
    # For librosa 0.11.0+, use keyword args:
    return librosa.effects.time_stretch(y=y, rate=rate)

def extract_augmented_features(file_path, n_mfcc=40):
    try:
        print(f"\nProcessing file: {file_path}")
        # Load audio at 22050 Hz
        y, sr = librosa.load(file_path, sr=22050)

        # 1) Original MFCC
        mfcc_orig = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)

        # 2) Pitch-shifted up by 2 semitones
        print(" -> pitch_shift_audio called with steps=2")
        y_pitch_up = pitch_shift_audio(y, sr, 2)
        mfcc_pitch_up = np.mean(librosa.feature.mfcc(y=y_pitch_up, sr=sr, n_mfcc=n_mfcc), axis=1)

        # 3) Time-stretched (rate=0.9 => slower)
        print(" -> time_stretch_audio called with rate=0.9")
        y_slow = time_stretch_audio(y, 0.9)
        mfcc_slow = np.mean(librosa.feature.mfcc(y=y_slow, sr=sr, n_mfcc=n_mfcc), axis=1)

        return [mfcc_orig, mfcc_pitch_up, mfcc_slow]
    except Exception as e:
        print(f"❌ Error processing {file_path} with augmentation: {e}")
        return []

if __name__ == "__main__":
    # Absolute path to your RAVDESS dataset on E: drive
    dataset_path = r"E:\emosense\archive (1)"

    data = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Example: 03-01-05-01-01-01-01.wav => parts[2] is emotion code
                parts = file.split('-')
                emotion_code = int(parts[2])

                # Extract original + augmented features
                feature_list = extract_augmented_features(file_path, n_mfcc=40)
                for feat in feature_list:
                    row = np.append(feat, emotion_code)
                    data.append(row)

    # Create DataFrame with 40 MFCC columns + 1 emotion column
    columns = [f"MFCC_{i+1}" for i in range(40)] + ["emotion"]
    df = pd.DataFrame(data, columns=columns)

    # Save as CSV in the same folder as this script (E:\emosense)
    df.to_csv("features_labels_augmented.csv", index=False)
    print("\n✅ Augmented feature extraction complete. Saved as 'features_labels_augmented.csv'")
