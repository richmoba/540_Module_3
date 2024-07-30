import streamlit as st
import http.client
import json
import pandas as pd 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Replace with your actual RapidAPI key
RAPIDAPI_KEY = 'f9795e5735mshdedf63bc56d7fcbp131f52jsn30ca7a3d4f15'
RAPIDAPI_HOST_IMDB8 = 'imdb8.p.rapidapi.com'
RAPIDAPI_HOST_IMDB188 = 'imdb188.p.rapidapi.com'

def get_imdb_data(title):
    url = "https://imdb8.p.rapidapi.com/title/find"
    headers = {
        'x-rapidapi-host': RAPIDAPI_HOST_IMDB8,
        'x-rapidapi-key': RAPIDAPI_KEY
    }
    params = {
        'q': title
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=(30, 60))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request error occurred: {req_err}")
    return None

def get_popular_movies():
    conn = http.client.HTTPSConnection(RAPIDAPI_HOST_IMDB188)

    payload = json.dumps({
        "country": {"anyPrimaryCountries": ["IN"]},
        "limit": 200,
        "releaseDate": {"releaseDateRange": {"end": "2029-12-31", "start": "2020-01-01"}},
        "userRatings": {"aggregateRatingRange": {"max": 10, "min": 6}, "ratingsCountRange": {"min": 1000}},
        "genre": {"allGenreIds": ["Action"]},
        "runtime": {"runtimeRangeMinutes": {"max": 120, "min": 0}}
    })

    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST_IMDB188,
        'Content-Type': "application/json"
    }

    conn.request("POST", "/api/v1/getPopularMovies", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))

def main():
    st.title("TV Show and Movie Recommendation System")
    
    # TV show search section
    st.subheader("TV Show Search")
    title = st.text_input("Enter a TV show title:")
    
    if st.button("Get TV Show Recommendations"):
        if title:
            imdb_data = get_imdb_data(title)
            if imdb_data:
                st.subheader("IMDb Data:")
                st.json(imdb_data)
            else:
                st.error("Failed to fetch data from IMDb.")
        else:
            st.warning("Please enter a TV show title.")
    
    # Popular movies section
    st.subheader("Popular Movies")
    if st.button("Get Popular Movies"):
        popular_movies = get_popular_movies()
        if popular_movies and 'data' in popular_movies and 'list' in popular_movies['data']:
            movies = popular_movies['data']['list']
            if movies:
                st.subheader("Popular Movies Data:")
                for movie in movies:
                    movie_title = movie.get('title', {}).get('titleText', {}).get('text', 'N/A')
                    release_year = movie.get('title', {}).get('releaseYear', {}).get('year', 'N/A')
                    rating = movie.get('title', {}).get('ratingsSummary', {}).get('aggregateRating', 'N/A')
                    vote_count = movie.get('title', {}).get('ratingsSummary', {}).get('voteCount', 'N/A')
                    image_url = movie.get('title', {}).get('primaryImage', {}).get('imageUrl', '')
                    
                    st.markdown(f"### {movie_title} ({release_year})")
                    st.markdown(f"**Rating:** {rating} ({vote_count} votes)")
                    if image_url:
                        st.image(image_url, width=200)
            else:
                st.error("No popular movies found.")
        else:
            st.error("Failed to fetch popular movies data.")

if __name__ == "__main__":
    main()

# This code snippet demonstrates how to build a TV show and movie recommendation system using Streamlit and the IMDb API. The system allows users to search for TV shows and get recommendations based on the entered title. It also provides a list of popular movies based on specific criteria such as country, release date, genre, user ratings, and runtime.