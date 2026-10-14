import mlflow
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/iris.csv")

X = df.iloc[:, 0:4]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment("Iris Classification")

with mlflow.start_run():

    n_estimators = 100

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)