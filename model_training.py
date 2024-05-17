import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle

def train_models(X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features):
    models_params = {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'regressor__alpha': [0.1, 1.0, 10.0]}),
        'Lasso': (Lasso(), {'regressor__alpha': [0.1, 1.0, 10.0]}),
        'DecisionTreeRegressor': (DecisionTreeRegressor(), {'regressor__max_depth': [10, 20, 30]}),
        'RandomForestRegressor': (RandomForestRegressor(), {'regressor__n_estimators': [50, 100, 200], 'regressor__max_depth': [10, 20, 30]}),
        'GradientBoostingRegressor': (GradientBoostingRegressor(), {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]}),
        'SVR': (SVR(), {'regressor__C': [0.1, 1.0, 10.0], 'regressor__gamma': [0.01, 0.1, 1.0]})
    }

    best_model = None
    best_mse = float('inf')

    for model_name, (model, params) in models_params.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        if params:
            grid_search = GridSearchCV(pipeline, params, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        if mse < best_mse:
            best_mse = mse
            best_model = model

        with mlflow.start_run():
            mlflow.log_param("model", model_name)
            mlflow.log_metric("mean_squared_error", mse)

    # Save the best Gradient Boosting model to a .pkl file
    if isinstance(best_model.named_steps['regressor'], GradientBoostingRegressor):
        with open('best_gradient_boosting_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print("Gradient Boosting model saved to best_gradient_boosting_model.pkl")

    return best_model
