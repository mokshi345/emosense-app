import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# ==================== Load the Dataset ====================
# Ensure that "features_labels_augmented.csv" is in the same folder as this script.
df = pd.read_csv("features_labels_augmented.csv")
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

# ==================== Prepare Features and Labels ====================
# X contains the 40 MFCC features, y contains the emotion labels.
X = df.drop(columns=["emotion"])
y = df["emotion"]

# Split the dataset: 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ==================== Train the Model ====================
# Initialize the RandomForest classifier with 100 trees.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ==================== Evaluate the Model ====================
# Predict on the test set.
y_pred = clf.predict(X_test)

# Calculate accuracy.
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

# Print the classification report.
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ==================== Save the Trained Model ====================
# Save the model using pickle so it can be loaded later in your app.
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("\n✅ Model saved as 'rf_model.pkl'.")
