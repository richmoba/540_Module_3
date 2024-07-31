import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
    # Create a mapping for user IDs and movie IDs
    user_id_map = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_map = {id: i for i, id in enumerate(movies['movieId'].unique())}
    
    num_users = len(user_id_map)
    num_items = len(movie_id_map)
    model = DeepModel(num_users, num_items)
    
    # Prepare the data
    df = pd.merge(ratings, movies, on='movieId')
    
    # Map the user and movie IDs
    df['user_idx'] = df['userId'].map(user_id_map)
    df['movie_idx'] = df['movieId'].map(movie_id_map)
    
    # Convert to PyTorch tensors
    users = torch.LongTensor(df['user_idx'].values)
    items = torch.LongTensor(df['movie_idx'].values)
    ratings = torch.FloatTensor(df['rating'].values)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    batch_size = 64
    for epoch in range(num_epochs):
        for i in range(0, len(users), batch_size):
            user_batch = users[i:i+batch_size]
            item_batch = items[i:i+batch_size]
            rating_batch = ratings[i:i+batch_size]
            
            # Forward pass
            predictions = model(user_batch, item_batch).squeeze()
            loss = criterion(predictions, rating_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Save model state and dataset info
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_items': num_items,
        'user_id_map': user_id_map,
        'movie_id_map': movie_id_map
    }, 'models/deep_model.pth')
    
    return model

def load_classical_model(path):
    return torch.load(path)

def load_deep_model(path):
    checkpoint = torch.load(path)
    num_users = checkpoint['num_users']
    num_items = checkpoint['num_items']
    model = DeepModel(num_users, num_items)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['user_id_map'], checkpoint['movie_id_map']

def get_recommendations(movie_title, model, movies, n=5, model_type='classical'):
    if model_type == 'classical':
        movie_idx = movies[movies['title'] == movie_title].index[0]
        similar_movies = model.get_recommendations(movie_idx, n)
        recommendations = list(zip(movies.iloc[similar_movies]['title'], model.similarity_matrix[movie_idx][similar_movies]))
    else:
        model, user_id_map, movie_id_map = model  # Unpack the model and mappings
        rev_movie_id_map = {v: k for k, v in movie_id_map.items()}
        movie_idx = movie_id_map[movies[movies['title'] == movie_title]['movieId'].iloc[0]]
        
        with torch.no_grad():
            users = torch.arange(model.user_embed.num_embeddings)
            items = torch.LongTensor([movie_idx] * model.user_embed.num_embeddings)
            ratings = model(users, items).squeeze()
        
        top_user_indices = ratings.argsort(descending=True)[:n]
        
        recommendations = []
        for user_idx in top_user_indices:
            # Get the original user ID
            user_id = list(user_id_map.keys())[list(user_id_map.values()).index(user_idx.item())]
            
            # Get all movie ratings for this user
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx.item()] * len(movie_id_map))
                movie_tensor = torch.arange(len(movie_id_map))
                user_ratings = model(user_tensor, movie_tensor).squeeze()
            
            # Get the top-rated movie for this user (excluding the input movie)
            top_movie_idx = user_ratings.argsort(descending=True)[1]  # [1] to get the second-best (first is the input movie)
            top_movie_id = rev_movie_id_map[top_movie_idx.item()]
            movie_title = movies[movies['movieId'] == top_movie_id]['title'].iloc[0]
            recommendations.append((movie_title, user_ratings[top_movie_idx].item()))
    
    return recommendations