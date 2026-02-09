import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
 
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def cosine_distance(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
        
    return 1 - (dot_product / (norm_a * norm_b))


class KNearestNeighbours:
    def __init__(self, k=3, dist=euclidean_distance):
        self.k = k
        self.dist = dist
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict_one(self, x):
        
        distances = []
        for i in range(len(self.X_train)):
            d = self.dist(self.X_train[i], x)
            distances.append((d, self.y_train[i]))
        
        
        distances.sort(key=lambda pair: pair[0])
        
        neighbors = distances[:self.k]
        labels = [lab for _, lab in neighbors]
 
        return max(set(labels), key=labels.count)
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct = 0
        wrong_indices = []
        
        for i in range(len(y)):
            if predictions[i] == y[i]:
                correct += 1
            else:
                wrong_indices.append(i)
                
        accuracy = (correct / len(y)) * 100
        return accuracy, wrong_indices
 

print("Loading Fashion MNIST...")
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

X_train = X_train_full[:2000]
y_train = y_train_full[:2000]
X_test  = X_test_full[:200]
y_test  = y_test_full[:200]
 
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test  = X_test.reshape(-1, 28*28)  / 255.0

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

 

print("Trying different values of K (Euclidean distance)")

for k in [1, 3, 5, 7]:
    knn = KNearestNeighbours(k=k, dist=euclidean_distance)
    knn.fit(X_train, y_train)
    
    acc, _ = knn.score(X_test, y_test)
    print(f"  k = {k:2d}   accuracy = {acc:.2f}%")
    
 

print("\n Comparing distance measures (k=3)")

knn_eu = KNearestNeighbours(k=3, dist=euclidean_distance)
knn_ma = KNearestNeighbours(k=3, dist=manhattan_distance)
knn_co = KNearestNeighbours(k=3, dist=cosine_distance)

knn_eu.fit(X_train, y_train)
knn_ma.fit(X_train, y_train)
knn_co.fit(X_train, y_train)

acc_eu, misclassified = knn_eu.score(X_test, y_test)
acc_ma, _ = knn_ma.score(X_test, y_test)
acc_co, _ = knn_co.score(X_test, y_test)

print(f"Euclidean    : {acc_eu:5.2f}%")
print(f"Manhattan    : {acc_ma:5.2f}%")
print(f"Cosine       : {acc_co:5.2f}%")
print(f"\nNumber of misclassified examples (Euclidean): {len(misclassified)} / {len(y_test)}")