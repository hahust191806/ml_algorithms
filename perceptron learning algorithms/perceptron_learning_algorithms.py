import numpy as np 


class Perceptron():
    def __init__(self, eta=0.01, n_iter: int = 10):
        self.eta = eta 
        self.n_iter = n_iter 
        
    # fit data 
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.error_ = []

        for _ in range(self.n_iter): 
            error = 0 
            for xi, target in zip(X, y):
                # calculate y^ 
                y_pred = self.predict(xi)
                # calculate gradient 
                gradient = self.eta * (target - y_pred)
                # update the weights 
                self.w = self.w + gradient * xi
                # update bias 
                self.b = self.b + gradient 
                # update error 
                error += int(gradient != 0.0)
            
            self.error_.append(error)
        return self 

    def net_input(self, X):
        return np.dot(X, self.w) + self.b  
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)         