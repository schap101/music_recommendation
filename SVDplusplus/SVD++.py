import os.path

from surprise import SVDpp
import pandas as pd
import numpy as np
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import Reader
from surprise import accuracy
from surprise import SVD
from surprise.model_selection import GridSearchCV
import Deezerdata

# importing the train and test data as pandas dataframe
train_data = pd.read_csv("../Data/train-prep.csv")
test_data = pd.read_csv("../Data/test.csv")
# drop index column
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# format the train_data corresponding to the reader's ('user, item, rating')
# data preparation
prepareData = {"itemID": list(train_data.genre_id),
               "userID": list(train_data.user_id),
               "rating": list(train_data.is_listened),
               }

test_data["is_listened"] = np.nan
prepareTestData = {"itemID": list(test_data.genre_id),
                   "userID": list(test_data.user_id),
                   "rating": list(test_data.is_listened),
                   }

# loading dict into dataframe
df = pd.DataFrame(prepareData)  # train data
dft = pd.DataFrame(prepareTestData)  # test data

# Creating a Reader
reader = Reader(rating_scale=(0, 1))

# Arranging dataframe
data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
testdata = Dataset.load_from_df(dft[["userID", "itemID", "rating"]], reader)

# build full trainset
trainset = data.build_full_trainset()

# Apply SVD algorithm and train it
algo = SVD()
algo.fit(trainset)





    # cross_validate(NormalPredictor(), data, cv=2)

    # kf = KFold(n_splits=3)

    # random = NormalPredictor()
    # SVDplusAglorithm = SVDpp()

    # for trainset, testset in kf.split(data):
    # train and test algorithm.
    #    random.fit(trainset)
    #    predictions = random.test(testset)

    # compute and print Root Mean Squared Error
    #    accuracy.rmse(predictions, verbose=True)

    # for trainset, testset in kf.split(data):
    # train and test SVDplusAlgorithm.
    #    SVDplusAglorithm.fit(trainset)
    #    predictions = SVDplusAglorithm.test(testset)

    # compute and print Root Mean Squared Error
    #accuracy.rmse(predictions, verbose=True)


# predic for testset
for row in range(0, testdata):
    algo.predict(row.userID, row.itemID)