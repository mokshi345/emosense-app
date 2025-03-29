import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 1) Load the augmented CSV file (created by feature_extraction_aug.py)
# It's in the same E:\emosense folder, so we use a relative path
df = pd.read_csv("features_labels_augmented.csv")
print("Dataset shape:", df.shape)
print("First 5 rows:\n", df.head())

# 2) Separate features and labels
X = df.drop(columns=["emotion"])
y = df["emotion"]

# 3) Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 4) Initialize and train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5) Predict on test set
y_pred = clf.predict(X_test)

# 6) Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8) Save the RandomForest model using pickle in the same folder
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n✅ Model saved as 'rf_model.pkl'.")
