from preprocess import load_data, split_data
from train import train_model
from evaluate import evaluate_model
from utils import get_feature_importance

# Load data
df = load_data("../data/train.csv")

# Split
X_train, X_test, y_train, y_test = split_data(df, "SalePrice")

# Train
model = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)

# Feature importance
feat_imp = get_feature_importance(model)
print(feat_imp.head(10))