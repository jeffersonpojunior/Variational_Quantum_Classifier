from sklearn.datasets import make_classification # dataset linearmente separável
from sklearn.datasets import make_moons # 2 meias-luas entrelaçadas
from sklearn.datasets import make_circles # 2 círculos concêntricos

X_classification, Y_classification = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    flip_y=0.0,
    random_state=42
)
X_moons, Y_moons = make_moons(n_samples=200, noise=0.1)
X_circles, Y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5)

class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == "linear":
            self.X_train = X_classification
            self.Y_train = Y_classification
        elif dataset_name == "nonlinear0":
            self.X_train = X_moons
            self.Y_train = Y_moons
        elif dataset_name == "nonlinear1":
            self.X_train = X_circles
            self.Y_train = Y_circles
            
    def X(self):
        return self.X_train
    def Y(self):
        return self.Y_train
    def getName(self):
        return self.dataset_name