import numpy as np 
from sklearn.datasets import make_blobs
from pathlib import Path 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def get_regression_dataset(n_centers: int = 5, n_samples: int = 500, train_size: float = 0.8, save_file: bool = False, file_name: str = None):
    # get dataset
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, random_state=42)
    # stardard dataset
    X = StandardScaler().fit_transform(X)
    # save file 
    if save_file == True: 
        # config path to dataset folder 
        datasets_path = Path('datasets')     
        np.savez(datasets_path / file_name, array1=X, array2=y)   
    return X, y

def plot_dataset(X_train, y_train):
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    hue=y_train,
                    palette="deep",
                    legend=None
                    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == '__main__':
    X_train, y_train = get_regression_dataset(train_size=0.8)
    plot_dataset(X_train, y_train)