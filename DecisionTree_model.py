from save_model import save_model_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

model_name="Decision Tree Classifier"

#Importing Dataset
df = pd.read_csv("heart.csv")

# Define features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split data into training (80%) and testing (20%) (ensure test_size and random_state are consistent accross model for relevant comparison)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree model
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = tree_clf.predict(X_test_scaled)

# Define CSV file path where you want to save the report
csv_file_path = 'model_comparison.csv'

# Save performance
save_model_report(model_name, y_test, y_pred, csv_file_path)
