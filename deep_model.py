import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

# Load the augmented CSV file.
df = pd.read_csv("features_labels_augmented.csv")
print("Dataset shape:", df.shape)
print("First 5 rows:\n", df.head())

# Separate features and labels.
X = df.drop(columns=["emotion"]).values  # MFCC features.
y = df["emotion"].values                # Emotion labels (1 to 8).

# Convert labels from 1..8 to 0..7 for one-hot encoding.
y = y - 1
y = to_categorical(y, num_classes=8)

# Split into training (80%) and testing (20%).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Scale features for better training performance.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the deep learning model.
model = Sequential()

# 1st layer: 128 neurons, ReLU activation
model.add(Dense(128, input_dim=40, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# 2nd layer: 64 neurons, ReLU activation
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# 3rd layer: 32 neurons, ReLU activation
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer: 8 neurons (for 8 possible emotions), softmax for probability
model.add(Dense(8, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train the model (50 epochs, batch size 32)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification Report & Confusion Matrix
# -----------------------------------------
# 1) Get predicted labels for X_test
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)   # Convert one-hot to integer
y_test_labels = np.argmax(y_test, axis=1)   # Convert one-hot to integer

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

print("Confusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Deep Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the trained model
model.save("emosense_model.h5")
print("Model saved as 'emosense_model.h5'.")

