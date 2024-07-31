import os
import pandas as pd
from scripts.build_features import preprocess_data
from scripts.model import train_classical_model, train_deep_model
import torch

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Process data
    preprocess_data('data', 'data/processed')

    # Load processed data
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')

    print("Training classical model...")
    classical_model = train_classical_model(ratings, movies)
    torch.save(classical_model, 'models/classical_model.pkl')
    print("Classical model trained and saved.")

    print("Training deep model...")
    train_deep_model(ratings, movies)
    print("Deep model trained and saved.")

if __name__ == '__main__':
    main()