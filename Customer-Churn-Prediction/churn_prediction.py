import pandas as pd

# Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Basic Information
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
import matplotlib.pyplot as plt

# Churn Distribution Chart
churn_counts = df["Churn"].value_counts()

plt.figure(figsize=(6,4))
churn_counts.plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("churn_distribution.png")

plt.show()
from sklearn.preprocessing import LabelEncoder

# Create a copy
df_ml = df.copy()

# Convert text columns into numbers
le = LabelEncoder()

for col in df_ml.columns:
    if df_ml[col].dtype == "object":
        df_ml[col] = le.fit_transform(df_ml[col])

print("\nEncoded Dataset:")
print(df_ml.head())
from sklearn.preprocessing import LabelEncoder

# Create copy
df_ml = df.copy()

# Encode all text columns
for col in df_ml.columns:
    try:
        df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))
    except:
        pass

# Remove customerID
df_ml = df_ml.drop("customerID", axis=1)

print("\nData Types:")
print(df_ml.dtypes.head(10))

print("\nFirst 5 Encoded Rows:")
print(df_ml.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Features (X) and Target (y)
X = df_ml.drop("Churn", axis=1)
y = df_ml["Churn"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")