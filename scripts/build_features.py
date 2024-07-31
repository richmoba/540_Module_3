# scripts/build_features.py

import pandas as pd # pandas is a popular data manipulation library
import os   # os is a module in the Python standard library

def preprocess_data(input_dir, output_dir): # Define a function to preprocess the data
    # Read the raw data
    ratings = pd.read_csv(os.path.join(input_dir, 'ratings.csv'))   # Load the ratings data
    movies = pd.read_csv(os.path.join(input_dir, 'movies.csv')) # Load the movies data

    # Merge ratings with movie titles
    df = pd.merge(ratings, movies, on='movieId')    # Merge the ratings and movies dataframes

    # Create user-item matrix
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)   # Create a user-item matrix

    # Save processed data
    user_item_matrix.to_csv(os.path.join(output_dir, 'user_item_matrix.csv'))   # Save the user-item matrix
    ratings.to_csv(os.path.join(output_dir, 'ratings_processed.csv'), index=False)  # Save the processed ratings data
    movies.to_csv(os.path.join(output_dir, 'movies_processed.csv'), index=False)    # Save the processed movies data

    print(f"Processed data saved to {output_dir}")  # Print a message to the console