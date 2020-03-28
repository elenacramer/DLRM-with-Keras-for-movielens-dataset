import pandas as pd
import zipfile
import numpy as np 
csv1='ml-latest-small/ratings.csv'
csv2='ml-latest-small/movies.csv'
def get_data_ratings(csv_ratings=csv1,csv_movies=csv2):
    zf = zipfile.ZipFile('/home/elena/Downloads/ml-latest-small.zip')
    # reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(zf.open(csv1), names=r_cols)
    m_cols=['movie_id', 'title', 'genre']
    movies = pd.read_csv(zf.open(csv2), names=m_cols)
    # merging ratings and movies
    #ratings=pd.merge(ratings,movies,on='movie_id')
    return pd.read_csv(zf.open(csv1), names=r_cols)
    
def get_all_data():
    # reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(zf.open('ml-latest-small/ratings.csv'), names=r_cols)
    m_cols=['movie_id', 'title', 'genre']
    movies = pd.read_csv(zf.open('ml-latest-small/movies.csv'), names=m_cols)
    # merging ratings and movies
    data=pd.merge(ratings,movies,on='movie_id')
    zz = zipfile.ZipFile('/home/elena/Downloads/ml-100k.zip')
    # reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(zz.open('ml-100k/u.user'), sep='|', names=u_cols,encoding='latin-1')
    data=pd.merge(users,data, on='user_id')
    return data

def train_test_data(ratings):
    unique_movies = ratings.movie_id.unique() # returns a np array
    movie_to_index = {old: new for new, old in enumerate(unique_movies)} # indexing movie_id, tart at 0
    index_to_movie = {idx: movie for movie, idx in movie_to_index.items()}
    new_movies = ratings.movie_id.map(movie_to_index) # replaces movie_id with coresp. index
    ratings['movie_index']=new_movies

    train=pd.read_pickle('/home/elena/Downloads/traindata.pkl')
    test=pd.read_pickle('/home/elena/Downloads/testdata.pkl')
    train['movie_index']=train.movie_id.map(movie_to_index)
    test['movie_index']=test.movie_id.map(movie_to_index)
    return (train,test)