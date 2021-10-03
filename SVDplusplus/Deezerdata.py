import pandas as pd
import numpy as np


# all the data preparation is done here. Splitting the train data in 5 folds

class dataPrep:
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

