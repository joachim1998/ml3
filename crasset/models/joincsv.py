import pandas as pd 

train = pd.read_csv('Data/data_train.csv',dtype=object, encoding='latin-1')
user = pd.read_csv('Data/data_user.csv',dtype=object, encoding='latin-1')
movie = pd.read_csv('Data/data_movie.csv',dtype=object, encoding='latin-1')
test = pd.read_csv('Data/data_test.csv',dtype=object, encoding='latin-1')
train_output = pd.read_csv('Data/output_train.csv',dtype=object, encoding='latin-1')
movie.drop(columns=['movie_title','IMDb_URL','release_date','video_release_date'], axis=1, inplace=True)
user.drop(['zip_code'],axis=1, inplace=True)

#One hot encoding
user = pd.get_dummies(user, columns = ['gender'])
user = pd.get_dummies(user, columns = ['occupation'])


#Concatenating train, user and movie dataframes
train_movie = pd.merge_ordered(train, movie, left_by="movie_id", how="outer").fillna("")
train_user_movie = pd.merge_ordered(train_movie, user, left_by="user_id", how="outer").fillna("")
train_user_movie[["user_id","movie_id"]] = train_user_movie[["user_id","movie_id"]].apply(pd.to_numeric)
train_user_movie.dropna(how='any', inplace=True)
train_user_movie = train_user_movie.sort_values(["user_id","movie_id"],axis=0)


train_user_movie.to_csv("Data/train_user_movie_mergeTEST.csv", index=False)

#Concatenating test, user and movie dataframe
test_movie = pd.merge_ordered(test, movie, left_by="movie_id", how="outer").fillna("")
test_user_movie = pd.merge_ordered(test_movie, user, left_by="user_id", how="outer").fillna("")

test_user_movie.dropna(how='any', inplace=True)
test_user_movie = test_user_movie.sort_values(["user_id","movie_id"],axis=0)

test_user_movie.to_csv("Data/test_user_movie_mergeTEST.csv", index=False)

