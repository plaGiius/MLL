from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load the MNIST dataset
digits = load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

# Train SVM with different kernels
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)
print("Accuracy of linear kernel:", svc_linear.score(X_test, y_test))

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
print("Accuracy of RBF kernel:", svc_rbf.score(X_test, y_test))

# Train MLP neural network
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)

print("Accuracy of MLP neural network:", mlp.score(X_test, y_test))
