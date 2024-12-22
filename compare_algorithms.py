import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the algorithms to compare
algorithms = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Dictionary to store the results
results = {}

# Train and test each algorithm
for name, algo in algorithms.items():
    algo.fit(X_train, y_train)
    y_pred = algo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print the results
print("Algorithm Comparison Results:")
for algo, accuracy in results.items():
    print(f"{algo}: {accuracy:.4f}")

# Plotting the results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
algorithms = [item[0] for item in sorted_results]
accuracies = [item[1] for item in sorted_results]

plt.figure(figsize=(10, 6))
plt.barh(algorithms, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Algorithm Comparison')
plt.xlim([0, 1])
plt.tight_layout()
plt.show()
