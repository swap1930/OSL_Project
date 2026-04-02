import pandas as pd

from src.preprocessing import preprocess
from src.train_model import train


def main():
    # Load the data
    df = pd.read_csv("data/raw/ai4i2020.csv")

    # Preprocess the data
    X, y = preprocess(df)

    # Train the model
    train(X, y)

    print("Model training completed successfully!")
    print("Model saved to model/model.pkl")
    print("Scaler saved to model/scaler.pkl")


if __name__ == "__main__":
    main()
