# opencode-task-1
 Project Title:
K-Nearest Neighbours (KNN) from Scratch using Fashion-MNIST

Author:
Disha S [125EE0110]

Description:
This project implements a distance-based classification algorithm (K-Nearest
Neighbours) completely from scratch using Python and NumPy. No machine learning
libraries are used for the KNN implementation. The aim of this task is to
understand how similarity between data points is used for prediction.

Dataset:
Fashion-MNIST dataset is used. It contains 28x28 grayscale images of clothing
items belonging to 10 different classes.

To reduce runtime, a smaller subset of the dataset is used:
- 2000 training samples
- 200 test samples

Files Included:
1. knn.py
   - Main implementation file
   - Contains distance functions, KNN class, experiments, and evaluation

2. report.txt
   - Explains the approach, experiments, observations, and conclusions

How to Run:
1. Make sure Python is installed
2. Required libraries:
   - numpy
   - tensorflow (only for loading Fashion-MNIST)
3. Run the file using:
   python knn.py

Implementation Details:
- Images are flattened from 28x28 to 784-dimensional vectors
- Pixel values are normalized to [0, 1]
- KNN is implemented using object-oriented programming
- Distance metrics implemented:
  - Euclidean Distance
  - Manhattan Distance
  - Cosine Distance

Experiments Performed:
- Effect of different values of K (1, 3, 5, 7)
- Comparison of distance metrics using K = 3
- Accuracy evaluation and misclassification analysis

Conclusion:
This project helped in understanding the working of distance-based classifiers,
the role of distance metrics, and the impact of K in KNN.
