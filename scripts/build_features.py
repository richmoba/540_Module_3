# scripts/build_features.py

import pandas as pd
import os

def preprocess_data(input_dir, output_dir):
    # Read the raw data
    ratings = pd.read_csv(os.path.join(input_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(input_dir, 'movies.csv'))

    # Merge ratings with movie titles
    df = pd.merge(ratings, movies, on='movieId')

    # Create user-item matrix
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Save processed data
    user_item_matrix.to_csv(os.path.join(output_dir, 'user_item_matrix.csv'))
    ratings.to_csv(os.path.join(output_dir, 'ratings_processed.csv'), index=False)
    movies.to_csv(os.path.join(output_dir, 'movies_processed.csv'), index=False)

    print(f"Processed data saved to {output_dir}")