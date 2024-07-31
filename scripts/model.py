import pandas as pd # pandas is a popular data manipulation library
import torch    # PyTorch is a popular deep learning library
import torch.nn as nn   # Neural network module in PyTorch
import torch.optim as optim # Optimization module in PyTorch
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity function from scikit-learn

class ClassicalModel:   # Define a class for the classical model
    def __init__(self, user_item_matrix):   # Constructor method
        self.user_item_matrix = user_item_matrix    # Store the user-item matrix
        self.similarity_matrix = cosine_similarity(user_item_matrix.T)  # Calculate the item-item similarity matrix
    
    def get_recommendations(self, movie_idx, n=5):  # Define a method to get movie recommendations
        similar_scores = self.similarity_matrix[movie_idx]  # Get the similarity scores for the input movie
        similar_movies = similar_scores.argsort()[::-1][1:n+1]  # Get the indices of the most similar movies
        return similar_movies   # Return the indices of the most similar movies

class DeepModel(nn.Module): # Define a class for the deep learning model
    def __init__(self, num_users, num_items, embed_dim=50):  # Constructor method
        super(DeepModel, self).__init__()   # Call the constructor of the parent class
        self.user_embed = nn.Embedding(num_users, embed_dim)    # User embedding layer
        self.item_embed = nn.Embedding(num_items, embed_dim)    # Item embedding layer
        self.fc1 = nn.Linear(embed_dim*2, 64)   # Fully connected layer 1
        self.fc2 = nn.Linear(64, 32)    # Fully connected layer 2
        self.fc3 = nn.Linear(32, 1) # Fully connected layer 3
        self.relu = nn.ReLU()   # ReLU activation function

    def forward(self, user, item):  # Forward method
        user_emb = self.user_embed(user)    # User embedding
        item_emb = self.item_embed(item)    # Item embedding
        x = torch.cat([user_emb, item_emb], dim=-1)  # Concatenate user and item embeddings
        x = self.relu(self.fc1(x))  # Pass through FC1 and ReLU
        x = self.relu(self.fc2(x))  # Pass through FC2 and ReLU
        return self.fc3(x)  # Return the output of FC3

def train_classical_model(ratings, movies):  # Define a function to train the classical model
    df = pd.merge(ratings, movies, on='movieId')    # Merge the ratings and movies dataframes
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)   # Create a user-item matrix
    return ClassicalModel(user_item_matrix)   # Return an instance of the ClassicalModel class

def train_deep_model(ratings, movies):  # Define a function to train the deep learning model
    # Create a mapping for user IDs and movie IDs   
    user_id_map = {id: i for i, id in enumerate(ratings['userId'].unique())}    # Create a mapping for user IDs
    movie_id_map = {id: i for i, id in enumerate(movies['movieId'].unique())}   # Create a mapping for movie IDs
    
    num_users = len(user_id_map)    # Number of users
    num_items = len(movie_id_map)   # Number of items
    model = DeepModel(num_users, num_items)  # Initialize the deep learning model
    
    # Prepare the data
    df = pd.merge(ratings, movies, on='movieId')    # Merge the ratings and movies dataframes
    
    # Map the user and movie IDs
    df['user_idx'] = df['userId'].map(user_id_map)  # Map user IDs to indices
    df['movie_idx'] = df['movieId'].map(movie_id_map)   # Map movie IDs to indices
    
    # Convert to PyTorch tensors
    users = torch.LongTensor(df['user_idx'].values)   # User indices
    items = torch.LongTensor(df['movie_idx'].values)    # Movie indices
    ratings = torch.FloatTensor(df['rating'].values)    # Ratings
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()    # Mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    # Training loop
    num_epochs = 5  # Number of epochs
    batch_size = 64 # Batch size
    for epoch in range(num_epochs): # Iterate over epochs
        for i in range(0, len(users), batch_size):  # Iterate over batches
            user_batch = users[i:i+batch_size]  # User indices batch
            item_batch = items[i:i+batch_size]  # Movie indices batch
            rating_batch = ratings[i:i+batch_size]  # Ratings batch
            
            # Forward pass
            predictions = model(user_batch, item_batch).squeeze()   # Forward pass
            loss = criterion(predictions, rating_batch) # Calculate the loss
            
            # Backward pass and optimize
            optimizer.zero_grad()   # Zero gradients
            loss.backward() # Backward pass
            optimizer.step()    # Optimize
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")   # Print the loss
    
    # Save model state and dataset info
    torch.save({    # Save the model state and dataset info
        'model_state_dict': model.state_dict(),  # Model state dictionary
        'num_users': num_users, # Number of users
        'num_items': num_items, # Number of items
        'user_id_map': user_id_map, # User ID mapping
        'movie_id_map': movie_id_map    # Movie ID mapping
    }, 'models/deep_model.pth') # Save the model to a file
    
    return model    # Return the trained model

def load_classical_model(path): # Define a function to load the classical model
    return torch.load(path)     # Load the model from the specified path

def load_deep_model(path):  # Define a function to load the deep learning model
    checkpoint = torch.load(path)   # Load the model checkpoint
    num_users = checkpoint['num_users'] # Number of users
    num_items = checkpoint['num_items'] # Number of items
    model = DeepModel(num_users, num_items) # Initialize the deep learning model
    model.load_state_dict(checkpoint['model_state_dict'])   # Load the model state dictionary
    return model, checkpoint['user_id_map'], checkpoint['movie_id_map'] # Return the model and mappings

def get_recommendations(movie_title, model, movies, n=5, model_type='classical'):   # Define a function to get movie recommendations
    if model_type == 'classical':   # If the model is classical
        movie_idx = movies[movies['title'] == movie_title].index[0] # Get the index of the input movie
        similar_movies = model.get_recommendations(movie_idx, n)    # Get similar movies
        recommendations = list(zip(movies.iloc[similar_movies]['title'], model.similarity_matrix[movie_idx][similar_movies]))   # Get movie titles and similarity scores
    else:   # If the model is deep learning
        model, user_id_map, movie_id_map = model  # Unpack the model and mappings
        rev_movie_id_map = {v: k for k, v in movie_id_map.items()}  # Reverse the movie ID mapping
        movie_idx = movie_id_map[movies[movies['title'] == movie_title]['movieId'].iloc[0]]   # Get the index of the input movie
        
        with torch.no_grad():   # Disable gradient tracking
            users = torch.arange(model.user_embed.num_embeddings)   # Get all user indices
            items = torch.LongTensor([movie_idx] * model.user_embed.num_embeddings)   # Get the movie index
            ratings = model(users, items).squeeze()  # Get ratings for all users
        
        top_user_indices = ratings.argsort(descending=True)[:n]  # Get the top user indices
        
        recommendations = []    # List to store recommendations
        for user_idx in top_user_indices:   # Iterate over top user indices
            # Get the original user ID      
            user_id = list(user_id_map.keys())[list(user_id_map.values()).index(user_idx.item())]   # Get the original user ID
            
            # Get all movie ratings for this user
            with torch.no_grad():   # Disable gradient tracking
                user_tensor = torch.LongTensor([user_idx.item()] * len(movie_id_map))   # User tensor
                movie_tensor = torch.arange(len(movie_id_map))  # Movie tensor
                user_ratings = model(user_tensor, movie_tensor).squeeze()   # Get ratings for all movies
            
            # Get the top-rated movie for this user (excluding the input movie)
            top_movie_idx = user_ratings.argsort(descending=True)[1]  # [1] to get the second-best (first is the input movie)
            top_movie_id = rev_movie_id_map[top_movie_idx.item()]   # Get the movie ID
            movie_title = movies[movies['movieId'] == top_movie_id]['title'].iloc[0]    # Get the movie title
            recommendations.append((movie_title, user_ratings[top_movie_idx].item()))   # Append the recommendation
    
    return recommendations  # Return the recommendations