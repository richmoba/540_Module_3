import os
import pandas as pd
from scripts.build_features import preprocess_data
from scripts.model import train_classical_model, train_deep_model
import torch


 

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)

    # Process data
    preprocess_data('data', 'data/processed')

    # Train models
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')

    classical_model = train_classical_model(ratings, movies)
    torch.save(classical_model, 'models/classical_model.pkl')

    deep_model = train_deep_model(ratings, movies)
    torch.save(deep_model.state_dict(), 'models/deep_model.pth')

if __name__ == '__main__':
    main()