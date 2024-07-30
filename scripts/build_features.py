import pandas as pd

def preprocess_data(input_dir, output_dir):
    ratings = pd.read_csv(f"{input_dir}/ratings.csv")
    movies = pd.read_csv(f"{input_dir}/movies.csv")
    
    # Merge ratings with movie titles
    df = pd.merge(ratings, movies, on='movieId')
    
    # Create user-item matrix
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Save processed data
    user_item_matrix.to_csv(f"{output_dir}/user_item_matrix.csv")