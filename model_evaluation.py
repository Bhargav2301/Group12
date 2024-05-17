import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def plot_residuals(model, X_test, y_test, preprocessor):
    X_test_preprocessed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_preprocessed)
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

def evaluate_models(best_model, X_test, y_test, preprocessor, categorical_features, numerical_features):
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    
    plot_feature_importances(best_model, feature_names)
    plot_residuals(best_model, X_test, y_test, preprocessor)
