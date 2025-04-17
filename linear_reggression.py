from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Linear Regression Model
class LinearModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.split_data()
        self.train_model()

    def split_data(self):
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def train_model(self):
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.y_train)
