import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/Personal_Loan.csv")
df.drop(['ID', 'ZIP Code'], axis=1, inplace=True)
df.rename(columns={'Personal Loan': 'Defaulter'}, inplace=True)
df['Experience'] = df['Experience'].apply(lambda x: max(x, 0))

# Define columns
numerical_cols = ['Age', 'Experience', 'Income', 'CCAvg']
categorical_cols = ['Family', 'Education', 'Mortgage', 'Securities Account',
                    'CD Account', 'Online', 'CreditCard']

# Preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Split data
X = df.drop('Defaulter', axis=1)
y = df['Defaulter']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
base_models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

# Set MLflow experiment
mlflow.set_experiment("BankDefaulter_Model_Comparison")
experiment = mlflow.get_experiment_by_name("BankDefaulter_Model_Comparison")
run_ids = []

# Train base models
for name, model in base_models.items():
    with mlflow.start_run(run_name=f"{name}_Base_Model") as run:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")
        run_ids.append(run.info.run_id)

# GridSearch with Bagging
param_grid = {
    'classifier__n_estimators': [10, 25],
    'classifier__max_samples': [0.8, 1.0],
    'classifier__estimator__max_depth': [3, 5]
}

# ✅ NEW (as per latest sklearn >=1.2)
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)

grid_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', bagging_model)
])
grid = GridSearchCV(grid_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

with mlflow.start_run(run_name="Bagging_Model_GridSearch") as run:
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(best_model, "model")
    run_ids.append(run.info.run_id)

# Identify best run
all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run = all_runs_df.sort_values(by="metrics.accuracy", ascending=False).iloc[0]
best_run_id = best_run.run_id
model_uri = f"runs:/{best_run_id}/model"

# Register best model
registered_model = mlflow.register_model(model_uri=model_uri, name="BankDefaulterBestModel")
print(f"\n✅ Best model registered: {registered_model.name}, version: {registered_model.version}")

# Save best model locally
final_model = mlflow.sklearn.load_model(model_uri)
joblib.dump(final_model, "../model/model.pkl")
print("📦 Final model saved to ../model/model.pkl")
