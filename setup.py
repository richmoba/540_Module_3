import os   # os is a module in the Python standard library   
import pandas as pd  # pandas is a popular data manipulation library
from scripts.build_features import preprocess_data  # Import the function we just created
from scripts.model import train_classical_model, train_deep_model   # Import the functions we just created
import torch    # PyTorch is a popular deep learning library

def main(): # Define the main function
    # Create necessary directories
    os.makedirs('models', exist_ok=True)    # Create a directory called 'models' if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)    # Create a directory called 'data/processed' if it doesn't exist

    # Process data
    preprocess_data('data', 'data/processed')   # Process the data and save it in the 'data/processed' directory

    # Load processed data
    ratings = pd.read_csv('data/ratings.csv')   # Load the processed ratings data
    movies = pd.read_csv('data/movies.csv') # Load the processed movies data

    print("Training classical model...")    # Print a message to the console
    classical_model = train_classical_model(ratings, movies)    # Train the classical model
    torch.save(classical_model, 'models/classical_model.pkl')   # Save the trained model to a file
    print("Classical model trained and saved.")   # Print a message to the console

    print("Training deep model...") # Print a message to the console
    train_deep_model(ratings, movies)   # Train the deep model
    print("Deep model trained and saved.")  # Print a message to the console

if __name__ == '__main__':  # If the script is run from the command line
    main()  # Call the main function