from prefect import flow, task
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@task
def load_dataset():

    print("Loading Iris Dataset...")

    df = pd.read_csv("iris(6).csv")

    print("Dataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    return df


@task
def preprocess_data(df):

    print("\nChecking Missing Values:")

    print(df.isnull().sum())

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    X = df[features]
    y = df["species"]

    print("\nFeatures:")
    print(features)

    print("\nInput Shape:")
    print(X.shape)

    print("\nTarget Shape:")
    print(y.shape)

    return X, y


@task
def train_model(X, y):

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Training Completed")

    print("Accuracy:", accuracy)

    return model, X_test, y_test


@task
def make_prediction(model, X_test, y_test):

    print("\nMaking Iris Prediction...")

    sample = X_test.iloc[[0]]

    actual = y_test.iloc[0]

    prediction = model.predict(sample)[0]

    print("\nInput Values:")
    print(sample.to_string(index=False))

    print("\nActual Species:")
    print(actual)

    print("\nPredicted Species:")
    print(prediction)

    return prediction


@flow
def iris_prediction_workflow():

    print("====================================")
    print("IRIS PREDICTION PREFECT WORKFLOW")
    print("====================================")

    df = load_dataset()

    X, y = preprocess_data(df)

    model, X_test, y_test = train_model(X, y)

    prediction = make_prediction(
        model,
        X_test,
        y_test
    )

    print("\n====================================")
    print("WORKFLOW COMPLETED")
    print("Final Prediction:", prediction)
    print("====================================")

    return prediction


if __name__ == "__main__":
    iris_prediction_workflow()
