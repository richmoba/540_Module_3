# Richmond Baker Movie Recommendation System using Streamlit Module 3
# Import the required libraries

import streamlit as st  # Streamlit version 0.89.0
import pandas as pd # Pandas version 1.3.3
from sklearn.metrics.pairwise import cosine_similarity  # Scikit-learn version 1.0
import tensorflow as tf  # TensorFlow version 2.6.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from sklearn.model_selection import train_test_split

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

# Classical ML Approach: Compute cosine similarity between movies
try:        
    similarity_matrix = cosine_similarity(ratings.T)    # Compute the cosine similarity between movies
    similarity_df = pd.DataFrame(similarity_matrix, index=ratings.columns, columns=ratings.columns) # Create a DataFrame from the similarity matrix
except Exception as e:  # Handle any exceptions
    st.error(f"Error computing cosine similarity: {e}") # Display an error message

# Function to get movie recommendations using classical ML approach
def get_recommendations_classical(movie_title, similarity_df, num_recommendations=5):             
    if movie_title in similarity_df.columns:    # Check if the movie is in the DataFrame
        similar_movies = similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations+1]   # Get the most similar movies
        return similar_movies   # Return the similar movies
    else:       # If the movie is not found
        return "Movie not found in dataset"   # Return an error message

# Prepare data for deep learning approach
user_ids = df['userId'].unique()
movie_ids = df['movieId'].unique()
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
df['userId'] = df['userId'].map(user_to_index)
df['movieId'] = df['movieId'].map(movie_to_index)

# Split the data
X = df[['userId', 'movieId']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
num_users = len(user_ids)
num_movies = len(movie_ids)
embedding_size = 50

model = Sequential()
model.add(Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1))
model.add(Flatten())
model.add(Dense(embedding_size, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train['userId'], X_train['movieId']], y_train, epochs=5, batch_size=32, validation_split=0.1)

# Function to get movie recommendations using deep learning approach
def get_recommendations_nn(user_id, movie_title, df_movies, num_recommendations=5):
    movie_id = df_movies[df_movies['title'] == movie_title]['movieId'].values
    if len(movie_id) == 0:
        return "Movie not found in dataset"
    movie_id = movie_id[0]
    user_index = user_to_index[user_id]
    movie_index = movie_to_index[movie_id]
    predicted_ratings = []
    for movie in movie_ids:
        predicted_ratings.append(model.predict([[user_index], [movie]]))
    predicted_ratings = [rating[0][0] for rating in predicted_ratings]
    top_movies_indices = sorted(range(len(predicted_ratings)), key=lambda i: predicted_ratings[i], reverse=True)[:num_recommendations]
    top_movies = [movie_ids[i] for i in top_movies_indices]
    top_movies_titles = df_movies[df_movies['movieId'].isin(top_movies)]['title'].values
    return top_movies_titles

# Streamlit interface
st.title("Movie Recommendation System") # Display the title of the app

# Classical ML Approach
st.subheader("Classical ML Approach")
movie_title = st.text_input("Enter a Movie Title (e.g., Toy Story (1995)):")    # Text input for the movie title

if st.button("Get Similar Movies (Classical ML)"): # Button to get similar movies
    if movie_title: # If a movie title is entered
        recommendations = get_recommendations_classical(movie_title, similarity_df)   # Get movie recommendations
        if isinstance(recommendations, str):    # If the recommendations are an error message
            st.error(recommendations)   # Display the error message
        else:   # If recommendations are found
            st.subheader(f"Recommendations for '{movie_title}':")   # Display the recommendations
            for title, score in recommendations.items():    # Iterate through the recommendations
                st.write(f"{title} (Similarity score: {score:.2f})")    # Display the movie title and similarity score
    else:   # If no movie title is entered
        st.warning("Please enter a movie title.")   # Display a warning message

# Deep Learning Approach
st.subheader("Deep Learning Approach")
user_id = st.number_input("Enter User ID:", min_value=1, max_value=len(user_ids))    # Numeric input for the user ID
movie_title_nn = st.text_input("Enter a Movie Title for Deep Learning (e.g., Toy Story (1995)):")    # Text input for the movie title

if st.button("Get Similar Movies (Deep Learning)"): # Button to get similar movies
    if movie_title_nn: # If a movie title is entered
        recommendations_nn = get_recommendations_nn(user_id, movie_title_nn, df_movies)   # Get movie recommendations
        if isinstance(recommendations_nn, str):    # If the recommendations are an error message
            st.error(recommendations_nn)   # Display the error message
        else:   # If recommendations are found
            st.subheader(f"Recommendations for '{movie_title_nn}':")   # Display the recommendations
            for title in recommendations_nn:    # Iterate through the recommendations
                st.write(f"{title}")    # Display the movie title
    else:   # If no movie title is entered
        st.warning("Please enter a movie title.")   # Display a warning message
