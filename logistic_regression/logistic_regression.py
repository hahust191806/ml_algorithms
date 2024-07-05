import numpy as np 


class LogisticRegression():
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None 
        self.b = None 
        
    # calculate loss 
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # compute loss 
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1, y2)
    
    # feed forwar process 
    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.b
        A = self._sigmoid(z)
    
    # fit data ~ training 
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # init parameters 
        self.weights = np.zeros(n_features)
        self.b = 0 
        
        # gradient descent 
        for _ in range(self.n_iters):
            y_pred = self.feed_forward(X)
            dz = y_pred - y
            
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(y_pred - y)
            # update weights 
            self.weights -= self.lr * dw
            self.b -= self.lr * db 
            
    # predict 
    def predict(self, X):
        y_hat = np.dot(X, self.weights) + self.bias 
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
                