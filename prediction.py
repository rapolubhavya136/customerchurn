import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler



# Load your dataset
data = pd.read_csv('churn_data (1).csv')


# Drop non-useful columns
data = data.drop(['customer_ID'], axis=1)


# Encode categorical variables
label_encoder = LabelEncoder()

for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])


# Separate features and target
X = data.drop('churn', axis=1)
y = data['churn']


# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)


# Predictions
y_pred = rf_classifier.predict(X_test)


# Accuracy & Reports
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)


# Feature Importance Plot
feature_importances = rf_classifier.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)


plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# Cross-Validation Accuracy
cv_scores = cross_val_score(
    rf_classifier,
    X_scaled,
    y,
    cv=5
)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average CV accuracy: {cv_scores.mean():.2f}")