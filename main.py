import streamlit as st
import pandas as pd
from scripts.model import load_classical_model, load_deep_model, get_recommendations

st.title("Movie Recommendation System")

# Load models
@st.cache_resource
def load_models():
    classical_model = load_classical_model('models/classical_model.pkl')
    deep_model = load_deep_model('models/deep_model.pth')
    return classical_model, deep_model

classical_model, deep_model = load_models()

# Load movie data
@st.cache_data
def load_movie_data():
    return pd.read_csv('data/movies.csv')

movies = load_movie_data()

# Sidebar with movie list
st.sidebar.title("Movies in Database")
movie_list = movies['title'].tolist()
selected_movie = st.sidebar.selectbox("Select a movie", movie_list)

# Main content
st.header("Get Movie Recommendations")

# Sample search
st.subheader("Sample Search")
sample_movie = "Toy Story (1995)"  # You can change this to any movie in your database
st.write(f"Try searching for: {sample_movie}")

# User input
movie_title = st.text_input("Enter a Movie Title:", value=selected_movie)
model_choice = st.radio("Choose Model:", ("Classical ML", "Deep Learning"))

if st.button("Get Similar Movies"):
    if movie_title:
        if movie_title not in movies['title'].values:
            st.error("Movie not found in the database. Please check the spelling or try another movie.")
        else:
            if model_choice == "Classical ML":
                recommendations = get_recommendations(movie_title, classical_model, movies, model_type='classical')
            else:
                recommendations = get_recommendations(movie_title, deep_model, movies, model_type='deep')

            st.subheader(f"Recommendations for '{movie_title}':")
            for title, score in recommendations:
                st.write(f"{title} (Similarity score: {score:.2f})")
    else:
        st.warning("Please enter a movie title.")

# Display movie database
st.header("Movie Database")
movies_per_page = 50
num_pages = len(movies) // movies_per_page + (1 if len(movies) % movies_per_page > 0 else 0)
page = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
start_idx = (page - 1) * movies_per_page
end_idx = start_idx + movies_per_page

st.table(movies.iloc[start_idx:end_idx][['title', 'genres']])
st.write(f"Showing page {page} of {num_pages}")