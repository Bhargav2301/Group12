from dill import load
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models
from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

def main():
    # Define categorical and numerical features
    categorical_features = ['Channel', 'Language', 'Partnered', 'Mature']
    numerical_features = ['WatchTimeMinutes', 'StreamTimeMinutes', 'PeakViewers', 'Followers', 'FollowersGained', 'ViewsGained']
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(categorical_features, numerical_features)

    # Train models
    best_model = train_models(X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features)

    # Evaluate models
    evaluate_models(best_model, X_test, y_test, preprocessor, categorical_features, numerical_features)

if __name__ == "__main__":
    main()
