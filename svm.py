import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# 1. Load data
np.random.seed(12345)

# Change to correct path if needed
dname = ''
#%cd DATA  # Uncomment if needed to change directory
str0 = "_XGB_24.dat"
fnamex = 'DATA/x' + str0
fnamey = 'DATA/y' + str0

x = np.loadtxt(dname + fnamex, delimiter=" ", dtype=float)[:, :2] / 50
y = np.loadtxt(dname + fnamey).astype(int)

N, L = len(x), x.shape[1]
print(f"N={N}, L={L}")

# --- Add Gaussian noise to features ---
noise_level = 3.0 / 50.  # Adjust noise level here
noise = np.random.normal(loc=0.0, scale=noise_level, size=x.shape)
x_noisy = x + noise

# Plot noisy dataset to check noise level
plt.figure(figsize=(6, 5))
plt.scatter(x_noisy[:, 0],
            x_noisy[:, 1],
            c=y,
            cmap="viridis",
            edgecolors="k",
            alpha=0.6)
plt.title(f"Noisy Dataset (Noise std dev = {noise_level})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# 2. Split data (use noisy data now)
X_train, X_val, y_train, y_val = train_test_split(x_noisy,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=42)

# 3. Define log-space grid for SVM hyperparameters
log_C_vals = np.linspace(np.log(1e-2), np.log(1e4), 6)  # C from 0.01 to 100
log_gamma_vals = np.linspace(np.log(1e-3), np.log(1e4),
                             6)  # gamma from 0.001 to 1
C_mesh, gamma_mesh = np.meshgrid(log_C_vals, log_gamma_vals)

# 4. Grid search over C and gamma
Z = np.zeros_like(C_mesh)
best_acc = 0
best_params = {'C': None, 'gamma': None}

for i in range(C_mesh.shape[0]):
    for j in range(C_mesh.shape[1]):
        C = np.exp(C_mesh[i, j])
        gamma = np.exp(gamma_mesh[i, j])
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        Z[i, j] = acc

        if acc > best_acc:
            best_acc = acc
            best_params['C'] = C
            best_params['gamma'] = gamma

print(f"\nBest Accuracy: {best_acc:.4f}")
print(
    f"Best Parameters: C = {best_params['C']:.4f}, gamma = {best_params['gamma']:.4f}"
)

# 5. Plot accuracy surface
plt.figure(figsize=(6, 5))
cp = plt.contourf(C_mesh, gamma_mesh, Z, levels=20, cmap="viridis")
plt.colorbar(cp, label="Validation Accuracy")
plt.xlabel("log(C)")
plt.ylabel("log(gamma)")
plt.title("SVM Validation Accuracy Landscape")
plt.tight_layout()
plt.show()

# 6. Train best model
clf_best = svm.SVC(C=best_params['C'], gamma=best_params['gamma'])
clf_best.fit(X_train, y_train)
y_pred = clf_best.predict(X_val)

# 7. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
plt.title("Confusion Matrix for Best SVM")
plt.tight_layout()
plt.show()


# 8. Decision Boundary (for 2D input only)
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
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


plot_decision_boundary(clf_best,
                       X_val,
                       y_val,
                       title="SVM Decision Boundary on Validation Set")
