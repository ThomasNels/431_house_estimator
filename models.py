import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# Linear Regression Model
class LinearModel:
    def __init__(self):
        self.split_data()

    def split_data(self):
        # use pd.read_csv to read in files, create dataframe
        # define features and label columns

        # used if there are columns with string variables
        # self.encoder = OneHotEncoder(sparse_output=False)

        # define self.X and self.y based on feature labels

        # may get rid of validation depending on project format
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def train_model(self):
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.y_train)

    def test_accuracy(self):
        self.train_model()
        predictions = self.model.predict(self.X_test)
        return mean_squared_error(self.y_test, predictions)

    def prediction(self, X):
        # if we use encoding then we need to add stuff here, we can also get rid of function depending on requirements
        return self.model.predict(X)