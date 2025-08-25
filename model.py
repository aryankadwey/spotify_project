# Popularity Prediction Model for Spotify Songs

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 2: Load Dataset
df = pd.read_csv("tracks.csv")

# Step 3: Data Preprocessing
# Drop unnecessary columns
df = df.drop(columns=['track_id', 'album_name', 'release_date'], errors='ignore')

# Drop missing values
df = df.dropna()

# Scale selected features
scaler = MinMaxScaler()
df[['danceability', 'energy', 'tempo', 'valence']] = scaler.fit_transform(
    df[['danceability', 'energy', 'tempo', 'valence']]
)

# Create binary target variable
threshold = 60
df['hit'] = df['popularity'].apply(lambda x: 1 if x > threshold else 0)

# Drop the original popularity column
df = df.drop(columns=['popularity'])

# Step 4: Split Data into Training and Testing Sets
X = df.drop(columns=['hit'])
X = X.select_dtypes(include=np.number)  # Keep only numeric columns
y = df['hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: Save the Model
joblib.dump(model, "popularity_prediction_model.pkl")
print("Model saved as popularity_prediction_model.pkl")





# df = pd.read_csv("tracks.csv")
# df = df.dropna()
# df['hit'] = df['popularity'].apply(lambda x: 1 if x > 60 else 0)
# X = df.drop(columns=['hit', 'popularity'])
# X = X.select_dtypes(include=np.number)
# y = df['hit']

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train Random Forest Classifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
