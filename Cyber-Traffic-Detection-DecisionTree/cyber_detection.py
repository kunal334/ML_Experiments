import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("basic_cyber_dataset.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Size")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Info")
df.info()

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df["label"].value_counts())

# Bar Graph
df["label"].value_counts().plot(kind="bar")
plt.title("Normal vs Attack Traffic")
plt.xlabel("Traffic Type")
plt.ylabel("Count")
plt.show()

# NumPy Operations
duration_array = np.array(df["duration_ms"])

print("\nDuration Values")
print(duration_array)

print("\nMean Duration")
print(np.mean(duration_array))

print("\nMaximum Duration")
print(np.max(duration_array))

print("\nMinimum Duration")
print(np.min(duration_array))

# Feature Selection
X = df.drop(["connection_id", "label"], axis=1)
Y = df["label"]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)

# Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)

print("\nPredicted Values")
print(Y_pred)

print("\nActual Values")
print(Y_test.values)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("\nAccuracy")
print(accuracy)