# model_training/train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("../data/Personal_Loan.csv")
print("Initial Data Preview:\n", df.head())

# 2. Drop 'ID' and 'ZIP Code'
df.drop(['ID', 'ZIP Code'], axis=1, inplace=True)

# 3. Rename 'Personal Loan' to 'Defaulter'
df.rename(columns={'Personal Loan': 'Defaulter'}, inplace=True)

# 4. Define column groups
numerical_cols = ['Age', 'Experience', 'Income', 'CCAvg']
categorical_cols = ['Family', 'Education', 'Mortgage', 'Securities Account', 
                    'CD Account', 'Online', 'CreditCard']

# 5. Fix Experience values < 0
df['Experience'] = df['Experience'].apply(lambda x: max(x, 0))

# 6. Define preprocessing pipelines
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# 7. Split data
X = df.drop('Defaulter', axis=1)
y = df['Defaulter']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Define final BaggingClassifier
base_estimator = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(
    estimator=base_estimator,
    random_state=42
)

param_grid = {
    'classifier__n_estimators': [10, 25, 50],
    'classifier__max_samples': [0.6, 0.8, 1.0],
    'classifier__estimator__max_depth': [3, 5, 7]
}

# 9. Build complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', bagging_model)
])

# 10. Grid Search for best BaggingClassifier
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# 11. Evaluate best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Final BaggingClassifier Accuracy: {final_accuracy:.4f}")

# 12. Save model
joblib.dump(best_model, "../backend/model/model.pkl")
joblib.dump(preprocessor, "../backend/model/preprocessor.pkl")  # optional if needed separately

print("\n🎯 Final Bagging model saved as best_model.")
