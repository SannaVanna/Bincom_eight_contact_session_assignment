import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Step 2: Features & Target
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"].map({'Y': 1, 'N': 0})  # Encode target

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Preprocessing
numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 5: Full pipeline with Logistic Regression
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 6: Train model
pipeline.fit(X_train, y_train)

# Step 7: Predictions
y_pred = pipeline.predict(X_test)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Save predictions for all applicants
all_preds = pipeline.predict(X)
df_output = pd.DataFrame({
    "Loan_ID": df["Loan_ID"],
    "Loan_Status_Predicted": ["Y" if p == 1 else "N" for p in all_preds]
})
df_output.to_csv("logistic_regression_predictions.csv", index=False)
print("\nPredictions saved to logistic_regression_predictions.csv")

# Step 10: User input for testing
print("\n=== Test with Your Own Input ===")
user_data = {
    "Gender": input("Gender (Male/Female): "),
    "Married": input("Married (Yes/No): "),
    "Dependents": input("Dependents (0-3+): "),
    "Education": input("Education (Graduate/Not Graduate): "),
    "Self_Employed": input("Self_Employed (Yes/No): "),
    "ApplicantIncome": float(input("ApplicantIncome: ")),
    "CoapplicantIncome": float(input("CoapplicantIncome: ")),
    "LoanAmount": float(input("LoanAmount: ")),
    "Loan_Amount_Term": float(input("Loan_Amount_Term (e.g., 360): ")),
    "Credit_History": float(input("Credit_History (1.0 for good, 0.0 for bad): ")),
    "Property_Area": input("Property_Area (Urban/Semiurban/Rural): ")
}

# Convert to DataFrame for prediction
new_applicant = pd.DataFrame([user_data], columns=X.columns)

# Predict approval
pred = pipeline.predict(new_applicant)[0]
print("\nPrediction:", "Approved" if pred == 1 else"Rejected")