import pandas as pd
import numpy as np

from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


def get_predictions():
    data = pd.read_csv('https://drive.google.com/uc?export=download&id=1ibgtR07ULjL2Mf7YCiOl4mL1zxxDUNWW')

    y = data.SalePrice
    x = data.drop('SalePrice', axis=1)

    train_X, test_X, train_y, test_y = train_test_split(x, y)

    my_pipeline = make_pipeline(Imputer(), XGBRegressor())

    train_X = pd.get_dummies(train_X)
    test_X = pd.get_dummies(test_X)
    train_X, test_X = train_X.align(test_X, join='left', axis=1)

    my_pipeline.fit(train_X, train_y)
    predictions = my_pipeline.predict(test_X)

    print(np.mean(np.abs(predictions-test_y)))


def get_submission():
    train_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1ibgtR07ULjL2Mf7YCiOl4mL1zxxDUNWW')
    train_y = train_data.SalePrice
    train_x = train_data.drop('SalePrice', axis=1)

    test_x = pd.read_csv('https://drive.google.com/uc?export=download&id=1cmqIDhq9xn_5kv-ERbn-HkMVzg7tdG5v')

    my_pipeline = make_pipeline(Imputer(), XGBRegressor())

    train_x = pd.get_dummies(train_x)
    test_x = pd.get_dummies(test_x)
    train_x, test_x = train_x.align(test_x, join='left', axis=1)

    my_pipeline.fit(train_x, train_y)
    predictions = my_pipeline.predict(test_x)

    submission = pd.DataFrame(data=predictions, index=test_x.Id, columns=['SalePrice'])
    submission.to_csv('Housing_data_submission.csv')


if __name__ == '__main__':
    # get_predictions()
    get_submission()
