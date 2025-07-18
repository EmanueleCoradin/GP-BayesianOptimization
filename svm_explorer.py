import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SVMExplorer:
    def __init__(self, noise_std=3.0/50., test_size=0.3, seed=42):
        self.noise_std = noise_std
        self.test_size = test_size
        self.seed = seed
        self.best_model = None
        self.best_params = {}
        self.best_acc = None
        self.Z = None
        self.log_C_vals = None
        self.log_gamma_vals = None

    def load_data(self, x_path='./DATA/x_XGB_24.dat', y_path='./DATA/y_XGB_24.dat'):
        np.random.seed(self.seed)
        x = np.loadtxt(x_path, delimiter=" ", dtype=float)[:, :2] / 50
        y = np.loadtxt(y_path).astype(int)
        noise = np.random.normal(0.0, self.noise_std, size=x.shape)
        x_noisy = x + noise

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            x_noisy, y, test_size=self.test_size, random_state=self.seed
        )
        self.X_all = x_noisy
        self.y_all = y

    def run_grid_search(self, log_C_vals=None, log_gamma_vals=None):
        if log_C_vals is None:
            log_C_vals = np.linspace(np.log(1e-2), np.log(1e4), 6)
        if log_gamma_vals is None:
            log_gamma_vals = np.linspace(np.log(1e-3), np.log(1e4), 6)

        self.log_C_vals = log_C_vals
        self.log_gamma_vals = log_gamma_vals
        C_mesh, gamma_mesh = np.meshgrid(log_C_vals, log_gamma_vals)
        self.Z = np.zeros_like(C_mesh)
        self.best_acc = 0
        self.best_model = None
        self.best_params = {}

        for i in range(C_mesh.shape[0]):
            for j in range(C_mesh.shape[1]):
                C = np.exp(C_mesh[i, j])
                gamma = np.exp(gamma_mesh[i, j])
                clf = svm.SVC(C=C, gamma=gamma)
                clf.fit(self.X_train, self.y_train)
                acc = accuracy_score(self.y_val, clf.predict(self.X_val))
                self.Z[i, j] = acc
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_params = {'C': C, 'gamma': gamma}
                    self.best_model = clf

    def evaluate_best_model(self):
        if self.best_model is None:
            raise RuntimeError("No best model found. Run grid search or train first.")
        y_pred = self.best_model.predict(self.X_val)
        ConfusionMatrixDisplay.from_predictions(self.y_val, y_pred)
        plt.title("Confusion Matrix for Best SVM")
        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self, on_val=True, title="Decision Boundary"):
        if self.best_model is None:
            raise RuntimeError("No best model found. Run grid search or train first.")
        model = self.best_model
        X = self.X_val if on_val else self.X_train
        y = self.y_val if on_val else self.y_train
        h = 0.02
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap="viridis")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_accuracy_landscape(self):
        if self.Z is None:
            raise RuntimeError("No grid search results to plot.")
        C_mesh, gamma_mesh = np.meshgrid(self.log_C_vals, self.log_gamma_vals)
        plt.figure(figsize=(6, 5))
        cp = plt.contourf(C_mesh, gamma_mesh, self.Z, levels=20, cmap="viridis")
        plt.colorbar(cp, label="Validation Accuracy")
        plt.xlabel("log(C)")
        plt.ylabel("log(gamma)")
        plt.title("SVM Validation Accuracy Landscape")
        plt.tight_layout()
        plt.show()
    
    def evaluate_hyperparams(self, log_C, log_gamma):
        C = np.exp(log_C)
        gamma = np.exp(log_gamma)
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_val, clf.predict(self.X_val))
        return acc
