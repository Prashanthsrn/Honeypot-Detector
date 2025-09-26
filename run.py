# run.py

from src.preprocess import load_data, preprocess_data
from src.train import train_logistic_regression, train_random_forest, evaluate_model
from src.utils import print_label_distribution

# Paths
train_path = "Data/UNSW_NB15_training-set.csv"
test_path = "data/UNSW_NB15_testing-set.csv"

# Load data
train_df, test_df = load_data(train_path, test_path)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

# Check label distribution
print_label_distribution(y_train, "Train")
print_label_distribution(y_test, "Test")

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = train_logistic_regression(X_train, y_train)
print("Evaluating Logistic Regression:")
evaluate_model(lr_model, X_test, y_test)

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = train_random_forest(X_train, y_train)
print("Evaluating Random Forest:")
evaluate_model(rf_model, X_test, y_test)
