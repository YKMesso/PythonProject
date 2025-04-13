from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogicRegression

#Train model
def training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogicRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

