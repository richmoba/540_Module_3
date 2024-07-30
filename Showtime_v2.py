#Richmond Baker Movie Recommendation System using Streamlit Module 3
# Import the required libraries

import streamlit as st  # Streamlit version 0.89.0
import pandas as pd # Pandas version 1.3.3
from sklearn.metrics.pairwise import cosine_similarity  # Scikit-learn version 1.0

# Load the MovieLens dataset
@st.cache_data  # Cache the data to avoid loading it multiple times
def load_data():        
    df_ratings = pd.read_csv('ratings.csv') # Load ratings data
    df_movies = pd.read_csv('movies.csv')       # Load movies data
    return df_ratings, df_movies        

df_ratings, df_movies = load_data()         # Load the data         

# Check the integrity of the loaded data
st.write("Ratings DataFrame Shape:", df_ratings.shape)  # Display the shape of the ratings DataFrame
st.write("Movies DataFrame Shape:", df_movies.shape)    # Display the shape of the movies DataFrame

# Merge ratings with movie titles
df = pd.merge(df_ratings, df_movies, on='movieId')  # Merge the two DataFrames on the 'movieId' column
st.write("Merged DataFrame Shape:", df.shape)    # Display the shape of the merged DataFrame

# Identify and handle duplicate movie titles
duplicate_titles = df[df.duplicated(['title'], keep=False)] # Find duplicate movie titles
if not duplicate_titles.empty:  # If there are duplicates
    st.write("Duplicate Titles Found:") # Display a message
    st.write(duplicate_titles)  # Display the duplicate titles
    # Removing duplicates by averaging the ratings for each user-movie pair
    df = df.groupby(['userId', 'title']).agg({'rating': 'mean'}).reset_index()  # Group by user and movie, then average the ratings
    st.write("Duplicates handled. New DataFrame Shape:", df.shape)  # Display the new shape of the DataFrame

# Create a pivot table
try:            
    ratings = df.pivot(index='userId', columns='title', values='rating').fillna(0)  # Create a pivot table with users as rows, movies as columns, and ratings as values
    st.write("Pivot Table Shape:", ratings.shape)   # Display the shape of the pivot table
except Exception as e:  # Handle any exceptions
    st.error(f"Error creating pivot table: {e}")    # Display an error message

# Compute cosine similarity between movies
try:        
    similarity_matrix = cosine_similarity(ratings.T)    # Compute the cosine similarity between movies
    similarity_df = pd.DataFrame(similarity_matrix, index=ratings.columns, columns=ratings.columns) # Create a DataFrame from the similarity matrix
except Exception as e:  # Handle any exceptions
    st.error(f"Error computing cosine similarity: {e}") # Display an error message

# Function to get movie recommendations
def get_recommendations(movie_title, similarity_df, num_recommendations=5):             
    if movie_title in similarity_df.columns:    # Check if the movie is in the DataFrame
        similar_movies = similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations+1]   # Get the most similar movies
        return similar_movies   # Return the similar movies
    else:       # If the movie is not found
        return "Movie not found in dataset"   # Return an error message

# Streamlit interface
st.title("Movie Recommendation System") # Display the title of the app
movie_title = st.text_input("Enter a Movie Title (e.g., Toy Story (1995)):")    # Text input for the movie title

if st.button("Get Similar Movies"): # Button to get similar movies
    if movie_title: # If a movie title is entered
        recommendations = get_recommendations(movie_title, similarity_df)   # Get movie recommendations
        if isinstance(recommendations, str):    # If the recommendations are an error message
            st.error(recommendations)   # Display the error message
        else:   # If recommendations are found
            st.subheader(f"Recommendations for '{movie_title}':")   # Display the recommendations
            for title, score in recommendations.items():    # Iterate through the recommendations
                st.write(f"{title} (Similarity score: {score:.2f})")    # Display the movie title and similarity score
    else:   # If no movie title is entered
        st.warning("Please enter a movie title.")   # Display a warning message
