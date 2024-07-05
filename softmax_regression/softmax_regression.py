import numpy as np 


class SoftmaxRegression():
    def __init__(self, learning_rate, n_iters, n_classes) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.cls = n_classes
        self.weights = None 
        # self.b = None 
        
    def _softmax(self, score):
        exps = np.exp(score)
        exp_sums = np.sum(exps, axis=1, keepdims=True)
        return exps / exp_sums
    
    def compute_loss(self, y_pre, y):
        epsilon = 1e-6 
        loss = -np.mean(np.sum(np.log(y_pre + epsilon) * y, axis=1))
        return loss 
    
    def feed_forward(self, X):
        score = np.dot(X, self.weights) + self.bias 
        return self._softmax(score)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, self.cls)) 
        self.b = np.zeros(self.cls)

        for _ in range(self.n_iters):
            y_pred = self.feed_forward(X) # -> y_pred.shape = (n_samples, n_classes)
            cost = y_pred - y 
            gradients = (1 / n_samples) * np.dot(X, cost)
            self.weights -= self.lr * gradients
        
    def predict(self, X):  
        y_pred = self.feed_forward(X)
        return np.argmax(y_pred)