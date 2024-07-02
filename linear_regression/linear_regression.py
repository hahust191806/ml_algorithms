import numpy as np 
from prepare_datasets import get_regression_dataset, plot_dataset
from sklearn.linear_model import LinearRegression 

class LinearRegressionV1: 
    # init function 
    def __init__(self, lr: float = 0.01, n_iters: int = 1000) -> None: 
        self.lr = lr 
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None 
        
    # fit data 
    def fit(self, X, y) -> None:
        num_samples, num_features = X.shape # X shape [N, f]
        self.weights = np.random.rand(num_features) # w shape [f, 1]
        self.bias = 0
        
        for i in range(self.n_iters):
            # y_pred shape should be N, 1 
            y_pred = np.dot(X, self.weights) + self.bias 
            # calculate grandient 
            dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)
            # update weight
            self.weights = self.weights - self.lr * dw 
            self.bias = self.bias - self.lr * db 

    # predict 
    def predict(self, X): 
        return np.dot(X, self.weights) + self.bias 
    
X_train, X_test, y_train, y_test = get_regression_dataset(n_samples=500)

# linear regression model with from scratch 
model = LinearRegressionV1(n_iters=1000)
# linear regression model with sklearn 
regr = LinearRegression()
# train model 
model.fit(X_train, y_train)
regr.fit(X_train, y_train) 

# In ra các trọng số (weights) của mô hình
print(f"Weights, Bias theo Sklearn có giá trị là: {regr.coef_}, {regr.intercept_}")
print(f'Weights, Bias theo mô hình tự define có giá trị là: {model.weights}, {model.bias}')