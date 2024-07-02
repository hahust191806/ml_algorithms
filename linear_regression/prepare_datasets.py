import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pathlib import Path 
import matplotlib.pyplot as plt 


def get_regression_dataset(n_samples: int = 500, train_size: float = 0.8, save_file: bool = False, file_name: str = None):
    # get dataset
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=15, random_state=4)
    # split train test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    # save file 
    if save_file == True: 
        # config path to dataset folder 
        datasets_path = Path('datasets')     
        np.savez(datasets_path / file_name, array1=X_train, array2=X_test, array3=y_train, array4=y_test)   
    return X_train, X_test, y_train, y_test

def plot_dataset(X_train, X_test, y_train, y_test):
    # Vẽ dữ liệu train và test
    plt.figure(figsize=(10, 6))
    # Dữ liệu train
    plt.scatter(X_train, y_train, color='blue', label='Train data')
    # Dữ liệu test
    plt.scatter(X_test, y_test, color='red', label='Test data')
    # Thêm tiêu đề và nhãn
    plt.title('Regression Data')
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.legend()

    # Hiển thị biểu đồ
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_regression_dataset(train_size=0.8)
    plot_dataset(X_train, X_test, y_train, y_test)