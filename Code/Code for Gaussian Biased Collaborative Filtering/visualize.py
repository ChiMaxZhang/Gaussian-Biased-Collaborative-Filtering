import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Change working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the predicted ratings data
ratings = pd.read_csv('predicted_ratings.csv')

# Pivot the data - rows as movies, columns as users, values as ratings
movie_ratings = ratings.pivot(index='movieId', columns='userId', values='predictedRating')


# Select 100 random movies
random_movies = np.random.choice(movie_ratings.index, 100, replace=False)
movie_ratings = movie_ratings.loc[random_movies]

# Scale the data
scaler = StandardScaler()
movie_ratings_scaled = scaler.fit_transform(movie_ratings)

# Perform PCA
pca = PCA(n_components=2)
movie_ratings_pca = pca.fit_transform(movie_ratings_scaled)

# Load the movie data
movie_data = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                         names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western'])

# Get the genres for each movie
genre_columns = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

def get_genres(row):
    return ", ".join([genre for genre, is_present in row[genre_columns].items() if is_present])

movie_data['genres'] = movie_data.apply(get_genres, axis=1)

# Create a dictionary from movie_id to a tuple of (movie_title, genre)
movie_dict = pd.Series(list(zip(movie_data.movie_title.values, movie_data.genres.values)), index=movie_data.movie_id).to_dict()

# Plot the results
plt.figure(figsize=(10, 10))
plt.scatter(movie_ratings_pca[:, 0], movie_ratings_pca[:, 1])

# Add labels
for i, movie_id in enumerate(movie_ratings.index):
    # Replace movie_id with movie_title and genre from the dictionary
    movie_title, genres = movie_dict.get(int(movie_id), (movie_id, 'Unknown'))
    plt.text(movie_ratings_pca[i, 0], movie_ratings_pca[i, 1], f'{genres}', rotation=45)

plt.show()

