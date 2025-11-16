import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("stroke_preprocessing/data_train.csv")
df_test  = pd.read_csv("stroke_preprocessing/data_test.csv")

X_train, y_train = df_train.drop('stroke', axis=1), df_train['stroke']
X_test,  y_test  = df_test.drop('stroke', axis=1),  df_test['stroke']

mlflow.set_experiment("stroke-prediction")

model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

with mlflow.start_run(run_name="stroke-prediction"):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Test Accuracy: {accuracy}")