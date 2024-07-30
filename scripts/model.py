import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

class ClassicalModel:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.similarity_matrix = cosine_similarity(user_item_matrix.T)
    
    def get_recommendations(self, movie_idx, n=5):
        similar_scores = self.similarity_matrix[movie_idx]
        similar_movies = similar_scores.argsort()[::-1][1:n+1]
        return similar_movies

class DeepModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=50):
        super(DeepModel, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.fc1 = nn.Linear(embed_dim*2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, user, item):
        user_emb = self.user_embed(user)
        item_emb = self.item_embed(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def train_classical_model(ratings, movies):
    df = pd.merge(ratings, movies, on='movieId')
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return ClassicalModel(user_item_matrix)

def train_deep_model(ratings, movies):
    num_users = ratings['userId'].nunique()
    num_items = ratings['movieId'].nunique()
    model = DeepModel(num_users, num_items)
    # In a real scenario, you would train this model here
    return model

def load_classical_model(path):
    return torch.load(path)

def load_deep_model(path):
    num_users = pd.read_csv('data/ratings.csv')['userId'].nunique()
    num_items = pd.read_csv('data/movies.csv')['movieId'].nunique()
    model = DeepModel(num_users, num_items)
    model.load_state_dict(torch.load(path))
    return model

def get_recommendations(movie_title, model, movies, n=5, model_type='classical'):
    movie_idx = movies[movies['title'] == movie_title].index[0]
    
    if model_type == 'classical':
        similar_movies = model.get_recommendations(movie_idx, n)
        recommendations = list(zip(movies.iloc[similar_movies]['title'], model.similarity_matrix[movie_idx][similar_movies]))
    else:
        # For simplicity, we're using the classical approach for the deep model as well
        # In a real scenario, you'd use the trained deep model to generate recommendations
        similar_movies = model.get_recommendations(movie_idx, n)
        recommendations = list(zip(movies.iloc[similar_movies]['title'], model.similarity_matrix[movie_idx][similar_movies]))
    
    return recommendations