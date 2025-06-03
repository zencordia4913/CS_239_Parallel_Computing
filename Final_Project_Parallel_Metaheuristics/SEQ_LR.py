
import numpy as np
import time
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the logistic regression training function
def train_logistic_regression(X, y, penalty, C, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    
    for i in range(num_iterations):
        linear_model = np.dot(X, weights)
        predictions = sigmoid(linear_model)
        
        # Compute the gradient
        errors = predictions - y
        gradient = np.dot(X.T, errors) / num_samples
        
        # Apply regularization
        if penalty == 'l1':
            reg_term = C * np.sign(weights)
        elif penalty == 'l2':
            reg_term = 2 * C * weights
        else:
            reg_term = 0
        
        weights -= learning_rate * (gradient + reg_term)
    
    return weights

# Define the function to predict using logistic regression
def predict(X, weights):
    linear_model = np.dot(X, weights)
    predictions = sigmoid(linear_model)
    return [1 if p >= 0.5 else 0 for p in predictions]

# Define the function to evaluate logistic regression on the test set
def evaluate_model(X_test, y_test, weights):
    y_pred = predict(X_test, weights)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, mcc, confusion_mat

# Main function to perform hyperparameter tuning
def main():
    process = psutil.Process()
    memory_start = process.memory_info().rss

    # Load the dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter search
    best_accuracy = 0
    best_train_accuracy = 0
    best_hyperparameters = None
    best_weights = None

    start_time = time.time()

    # Define hyperparameter ranges
    penalties = ['l1', 'l2']
    C_values = np.logspace(-1, 1, num=100)
    learning_rates = np.logspace(-3, 0, num=100)
    num_iterations = 1000
    function_evaluations = 0
    
    for penalty in penalties:
        for C in C_values:
            for learning_rate in learning_rates:
                function_evaluations += 1
                # Train the model
                weights = train_logistic_regression(X_train_scaled, y_train, penalty, C, learning_rate, num_iterations)
                
                train_predictions = predict(X_train_scaled, weights)
                train_accuracy = accuracy_score(y_train, train_predictions)

                # Evaluate the model
                accuracy, precision, recall, f1, mcc, confusion_mat = evaluate_model(X_test_scaled, y_test, weights)
                
                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy

                # Check if this is the best model so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = (penalty, C, learning_rate)
                    best_weights = weights

    end_time = time.time()
    memory_end = process.memory_info().rss
    memory_usage = (memory_end - memory_start) / (1024 * 1024)  # Convert to MB
    
    print("\nBest Hyperparameters found:")
    print("Penalty:", best_hyperparameters[0])
    print("C:", best_hyperparameters[1])
    print("Learning Rate:", best_hyperparameters[2])
    print("Train accuracy:", best_train_accuracy)
    print("\nBest Model Performance on Test Set:")
    print("Accuracy:", best_accuracy)
    accuracy, precision, recall, f1, mcc, confusion_mat = evaluate_model(X_test_scaled, y_test, best_weights)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("MCC:", mcc)
    print("Confusion Matrix:\n", confusion_mat)
    print("\nTotal time taken:", end_time - start_time, "seconds")
    print("Memory usage:", memory_usage, "MB")
    print("Number of function evaluations:", function_evaluations)

if __name__ == "__main__":
    main()
