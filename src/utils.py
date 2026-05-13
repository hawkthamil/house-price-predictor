import pandas as pd

def get_feature_importance(model):
    best_model = model.named_steps["model"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importance = best_model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return feat_imp