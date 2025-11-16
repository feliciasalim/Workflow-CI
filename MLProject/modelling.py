import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import os

# Set tracking URI to local mlruns directory
mlflow.set_tracking_uri("file:./mlruns")

df_train = pd.read_csv("stroke_preprocessing/data_train.csv")
df_test  = pd.read_csv("stroke_preprocessing/data_test.csv")

X_train, y_train = df_train.drop('stroke', axis=1), df_train['stroke']
X_test,  y_test  = df_test.drop('stroke', axis=1),  df_test['stroke']

# Create or get experiment
experiment = mlflow.set_experiment("stroke-prediction")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Experiment Name: {experiment.name}")

model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

with mlflow.start_run():
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(
        model, 
        "model",
        input_example=X_train.iloc[:5]
    )
    
    print(f"Test Accuracy: {accuracy}")
    print(f"Model saved to: mlruns/{experiment.experiment_id}/{run_id}/artifacts/model")