import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Sudip-8345', repo_name='MLflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/MLflow-dagshub-demo.mlflow")


# Load data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set hyperparameters
max_depth = 3
n_estimators = 70

import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "Sudip-8345"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3a7b1bd52c535c004bd9b275516eae784255615d"
# Start MLflow run
with mlflow.start_run(experiment_id=1, run_name="RandomForest_Iris_Experiment"):
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Log parameters and metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(rf, "model")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confusion_matrix.png")
    mlflow.log_artifact("plots/confusion_matrix.png")

    mlflow.log_artifact(__file__)  # Log the script file

    mlflow.set_tag("author", "Sudip-8345")  # Set a tag for the run
    mlflow.set_tag("model_type", "RandomForest")  # Set a tag for the model type
    # mlflow.log_artifact("confusion_matrix.png")

    print("Model and metrics logged to MLflow.")
