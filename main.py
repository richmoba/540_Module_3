import streamlit as st  # Streamlit library
import pandas as pd # pandas is a popular data manipulation library
from scripts.model import load_classical_model, load_deep_model, get_recommendations    # Import the functions we just created

st.title("Movie Recommendation System") # Set the title of the web app

# Load models
@st.cache_resource  # Cache the results of this function
def load_models():  # Define a function to load the models
    classical_model = load_classical_model('models/classical_model.pkl')    # Load the classical model
    deep_model = load_deep_model('models/deep_model.pth')   # Load the deep model
    return classical_model, deep_model  # Return the loaded models

classical_model, deep_model = load_models() # Load the models

# Load movie data
@st.cache_data  # Cache the results of this function
def load_movie_data():  # Define a function to load the movie data
    return pd.read_csv('data/movies.csv')   # Load the movie data

movies = load_movie_data()  # Load the movie data

# Sidebar with movie list
st.sidebar.title("Movies in Database")  # Set the title of the sidebar
movie_list = movies['title'].tolist()   # Get a list of movie titles
selected_movie = st.sidebar.selectbox("Select a movie", movie_list)   # Create a dropdown to select a movie

# Main content
st.header("Get Movie Recommendations")  # Set the header of the main content

# Sample search
st.subheader("Sample Search")   # Set a subheader
sample_movie = "Toy Story (1995)"  # You can change this to any movie in your database
st.write(f"(Case Sensitive) Try searching for: {sample_movie}")  # Display a message to the user

# User input
movie_title = st.text_input("Enter a Movie Title:", value=selected_movie)       # Create a text input for the movie title
model_choice = st.radio("Choose Model:", ("Classical ML", "Deep Learning"))     # Create a radio button to choose the model

if st.button("Get Similar Movies"): # Create a button to get similar movies
    if movie_title: # If a movie title is entered
        if movie_title not in movies['title'].values:   # If the movie is not in the database
            st.error("Movie not found in the database. Please check the spelling or try another movie.")    # Display an error message
        else:   # If the movie is in the database
            if model_choice == "Classical ML":  # If the user chooses the classical model
                recommendations = get_recommendations(movie_title, classical_model, movies, model_type='classical')  # Get recommendations using the classical model
            else:   # If the user chooses the deep learning model
                recommendations = get_recommendations(movie_title, deep_model, movies, model_type='deep')   # Get recommendations using the deep learning

            st.subheader(f"Recommendations for '{movie_title}':")   # Display a subheader
            for title, score in recommendations:    # Iterate over the recommendations
                st.write(f"{title} (Similarity score: {score:.2f})")    # Display the movie title and similarity score
    else:   # If no movie title is entered
        st.warning("Please enter a movie title.")   # Display a warning message

# Display movie database
st.header("Movie Database")   # Set the header for the movie database
movies_per_page = 50    # Number of movies to display per page
num_pages = len(movies) // movies_per_page + (1 if len(movies) % movies_per_page > 0 else 0)    # Calculate the number of pages
page = st.number_input("Page", min_value=1, max_value=num_pages, value=1)   # Create a number input for the page number
start_idx = (page - 1) * movies_per_page    # Calculate the starting index
end_idx = start_idx + movies_per_page   # Calculate the ending index

st.table(movies.iloc[start_idx:end_idx][['title', 'genres']])   # Display the movies in a table
st.write(f"Showing page {page} of {num_pages}")   # Display the current page number and total number of pages