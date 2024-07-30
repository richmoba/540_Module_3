import streamlit as st
import pandas as pd
import torch
from scripts.model import load_classical_model, load_deep_model, get_recommendations

st.title("Movie Recommendation System")

 
# Load models
classical_model = load_classical_model('models/classical_model.pkl')
deep_model = load_deep_model('models/deep_model.pth')

# Load movie data
movies = pd.read_csv('data/movies.csv')

# User input
movie_title = st.text_input("Enter a Movie Title:")
model_choice = st.radio("Choose Model:", ("Classical ML", "Deep Learning"))

if st.button("Get Similar Movies"):
    if movie_title:
        if model_choice == "Classical ML":
            recommendations = get_recommendations(movie_title, classical_model, movies, model_type='classical')
        else:
            recommendations = get_recommendations(movie_title, deep_model, movies, model_type='deep')

        st.subheader(f"Recommendations for '{movie_title}':")
        for title, score in recommendations:
            st.write(f"{title} (Similarity score: {score:.2f})")
    else:
        st.warning("Please enter a movie title.")