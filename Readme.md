# Movie Recommendation System

This project implements a movie recommendation system using both classical machine learning and deep learning approaches. It uses the MovieLens dataset and provides recommendations through a Streamlit web interface.

## Features

- Classical ML approach using cosine similarity
- Deep Learning approach using a simple neural network (placeholder implementation)
- Streamlit web interface for user interaction

## Project Structure
├── README.md               <- The top-level README for developers using this project
├── requirements.txt        <- The requirements file for reproducing the analysis environment
├── setup.py                <- Script to set up project (process data, train models)
├── main.py                 <- Main script to run the Streamlit interface
├── scripts/                <- Source code for use in this project
│   ├── build_features.py   <- Script to turn raw data into features for modeling
│   └── model.py            <- Scripts to train models and then use trained models to make predictions
├── models/                 <- Trained and serialized models
├── data/                   <- Project data
│   ├── movies.csv          <- The movies dataset
│   ├── ratings.csv         <- The ratings dataset
│   └── processed/          <- The final, canonical data sets for modeling
└── .gitignore              <- List of files ignored by git
## Setup

1. Clone this repository:
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

3. Install the required dependencies:
pip install -r requirements.txt

4. Ensure that your `data` folder contains the following files:
- `movies.csv`: Contains movie information (movieId, title, genres)
- `ratings.csv`: Contains user ratings (userId, movieId, rating, timestamp)

5. Run the setup script to process data and train models:
python setup.py

## Usage

To start the Streamlit app, run:
streamlit run main.py

This will open a web interface where you can enter a movie title and choose between the classical ML and deep learning models to get movie recommendations.

## Data Format

The system expects the following data formats:

1. `movies.csv`:
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
5,Father of the Bride Part II (1995),Comedy
etc.

2. `ratings.csv`:
userId,movieId,rating,timestamp
1,1,4,964982703
1,3,4,964981247
1,6,4,964982224
Copy
## Limitations and Future Work

- The deep learning model is currently a placeholder and not fully implemented for recommendations.
- The system uses a small subset of the MovieLens dataset. For better results, consider using the full dataset.
- Future work could include implementing a more sophisticated deep learning model, improving the recommendation algorithm, and adding more features to the user interface.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

 