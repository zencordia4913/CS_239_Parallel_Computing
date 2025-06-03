
import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import psutil

# Kernel code
kernel_code = """
__global__ void evaluate_fitness(float *hyperparameters, float *data, float *labels, float *accuracies,
                                 int num_features, int num_samples, int pop_size, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < pop_size) {
        int penalty_idx = int(hyperparameters[4 * idx]);
        float C = hyperparameters[4 * idx + 1];
        int solver_idx = int(hyperparameters[4 * idx + 2]); 
        float l_r = hyperparameters[4 * idx + 3];

        float penalty = penalty_idx == 0 ? 0 : 1;

        float *weight = new float[num_features];
        float *gradients = new float[num_features];
        float initial_weight = idx * 0.01;

        for (int i = 0; i < num_features; i++) {
            weight[i] = initial_weight;
        }

        float learning_rate = l_r;
        int num_iterations = 100;

        for (int iter = 0; iter < num_iterations; iter++) {
            for (int j = 0; j < num_features; j++) {
                gradients[j] = 0.0;
            }
            for (int sample = 0; sample < num_samples; sample++) {
                float linear_combination = 0.0;
                for (int j = 0; j < num_features; j++) {
                    linear_combination += data[sample * num_features + j] * weight[j];
                }
                float prediction = 1.0 / (1.0 + exp(-linear_combination));
                float error = prediction - labels[sample];
                for (int j = 0; j < num_features; j++) {
                    gradients[j] += data[sample * num_features + j] * error;
                }
            }
            if (penalty == 0) {
                for (int j = 0; j < num_features; j++) {
                    float regularization_term = (weight[j] > 0) ? C : -C;
                    weight[j] -= learning_rate * (gradients[j] / num_samples + regularization_term / num_samples);
                }
            } else if (penalty == 1) {
                for (int j = 0; j < num_features; j++) {
                    weight[j] -= learning_rate * (gradients[j] / num_samples + 2 * C * weight[j]);
                }
            }
        }

        int correct_predictions = 0;
        for (int sample = 0; sample < num_samples; sample++) {
            float linear_combination = 0.0;
            for (int j = 0; j < num_features; j++) {
                linear_combination += data[sample * num_features + j] * weight[j];
            }
            float prediction = 1.0 / (1.0 + exp(-linear_combination));
            int predicted_label = (prediction >= 0.5) ? 1 : 0;
            if (predicted_label == int(labels[sample])) {
                correct_predictions += 1;
            }
        }
        accuracies[idx] = (float)correct_predictions / num_samples;

        delete[] weight;
        delete[] gradients;
    }
}
"""

# Define the tournament selection function
def tournament_selection(fitness, tournament_size):
    population_size = len(fitness)
    selected_parents = []
    
    for _ in range(population_size):
        tournament_indices = np.random.choice(population_size, size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(winner_index)
    
    return np.array(selected_parents)

def evaluate_test_set(X_test, y_test, X_train, y_train, best_hyperparameters, penalty_mod_mapping, Solver_mapping):
    penalty_idx = int(best_hyperparameters[0])
    C = best_hyperparameters[1]
    solver_idx = int(best_hyperparameters[2])
    learning_rate = best_hyperparameters[3]

    model = LogisticRegression(penalty=penalty_mod_mapping[penalty_idx],
                               C=C,
                               solver=Solver_mapping[solver_idx],
                               max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, mcc, confusion_mat

def main(x):
    process = psutil.Process()
    memory_start = process.memory_info().rss

    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    num_features = X_train_scaled.shape[1]
    num_samples = X_train_scaled.shape[0]

    hyperparameter_length = 4
    penalty_mod_mapping = ["l1", "l2"]
    penalty_mod = [0, 1]
    C_values = np.logspace(-1, 1, num=10*x)
    Solver_mapping = ["liblinear", "saga"]
    Solver = [0, 1]
    Learning_rates = np.logspace(-1, 0, num=10*x)

    pop_size = 256
    generations = 20
    mutation_rate = 0.12
    crossover_rate = 0.1
    tournament_size = 2

    population = np.empty((pop_size, hyperparameter_length), dtype=np.float32)
    for i in range(pop_size):
        population[i, 0] = np.random.randint(0, len(penalty_mod))
        population[i, 1] = C_values[np.random.randint(0, len(C_values))]
        population[i, 2] = np.random.randint(0, len(Solver))
        population[i, 3] = Learning_rates[np.random.randint(0, len(Learning_rates))]

    accuracies = np.zeros(pop_size, dtype=np.float32)
    X_train_scaled = X_train_scaled.flatten().astype(np.float32)

    X_train_gpu = drv.mem_alloc(X_train_scaled.nbytes)
    y_train_gpu = drv.mem_alloc(y_train.nbytes)
    hyperparameters_gpu = drv.mem_alloc(population.nbytes)
    accuracies_gpu = drv.mem_alloc(accuracies.nbytes)

    drv.memcpy_htod(X_train_gpu, X_train_scaled)
    drv.memcpy_htod(y_train_gpu, y_train)
    drv.memcpy_htod(hyperparameters_gpu, population)

    mod = SourceModule(kernel_code)
    evaluate_fitness = mod.get_function("evaluate_fitness")

    block_size = 256
    grid_size = (pop_size + block_size - 1) // block_size

    shared_memory_size = (2 * num_features) * 4

    # Metrics
    start_time = time.time()
    gpu_start = drv.Event()
    gpu_end = drv.Event()
    gpu_start.record()
    function_evaluations = 0
    
    for gen in range(generations):
        evaluate_fitness(hyperparameters_gpu, X_train_gpu, y_train_gpu, accuracies_gpu,
                         np.int32(num_features), np.int32(num_samples), np.int32(pop_size),
                         block=(block_size, 1, 1), grid=(grid_size, 1, 1), shared=shared_memory_size)
        function_evaluations += pop_size
        drv.memcpy_dtoh(accuracies, accuracies_gpu)

        selected_parents_indices = tournament_selection(accuracies, tournament_size)
        crossover_mask = np.random.rand(pop_size) < crossover_rate
        offspring = np.empty_like(population)
        offspring[crossover_mask] = population[selected_parents_indices[crossover_mask]]
        offspring[~crossover_mask] = population[np.random.choice(pop_size, size=np.sum(~crossover_mask), replace=True)]

        mutation_mask = np.random.rand(pop_size, hyperparameter_length) < mutation_rate
        for i in range(pop_size):
            if mutation_mask[i, 0]:
                offspring[i, 0] = np.random.randint(0, len(penalty_mod))
            if mutation_mask[i, 1]:
                offspring[i, 1] = C_values[np.random.randint(0, len(C_values))]
            if mutation_mask[i, 2]:
                offspring[i, 2] = np.random.randint(0, len(Solver))
            if mutation_mask[i, 3]:
                offspring[i, 3] = Learning_rates[np.random.randint(0, len(Learning_rates))]

        population = offspring.copy()
        drv.memcpy_htod(hyperparameters_gpu, population)
    
    gpu_end.record()
    gpu_end.synchronize()

    end_time = time.time()

    cpu_time = end_time - start_time
    gpu_time = gpu_start.time_till(gpu_end) * 1e-3

    best_index = np.argmax(accuracies)
    best_hyperparameters = population[best_index]
    best_fitness = accuracies[best_index]
    best_penalty_idx = int(best_hyperparameters[0])
    best_solution_idx = int(best_hyperparameters[2])

    print("\n---------Hyperparameter Search Summary----------")
    print("ML Model used: Logistic Regression")
    print("Dataset: Breast cancer dataset")
    print("Hyperparameter search method: Parallel Genetic Algorithm")
    print("Parallel strategy: 1C/RS/SPSS")
    print("\n")
    print("-----Best hyperparameter combination found-----")
    print("Penalty: ", penalty_mod_mapping[best_penalty_idx])
    print("Regularization constant, C: ", best_hyperparameters[1])
    print("Solution: ", Solver_mapping[best_solution_idx])
    print("Learning rate: ", best_hyperparameters[3])
    print("Best fitness (accuracy):", best_fitness)
    print("\n")
    print("-----Evaluating best hyperparameters on test set-----")
    accuracy, precision, recall, f1, mcc, confusion_mat = evaluate_test_set(X_test, y_test, X_train, y_train, best_hyperparameters, penalty_mod_mapping, Solver_mapping)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("MCC:", mcc)
    print("Confusion Matrix:\n", confusion_mat)
    print("\n")
    print("-----Parallel Search Strategy Performance-----")
    memory_end = process.memory_info().rss
    memory_usage = (memory_end - memory_start) / (1024 * 1024)
    print("Memory usage:", memory_usage, "MB")
    print("Total search time (CPU):", cpu_time, "seconds")
    print("Total search time (GPU):", gpu_time, "seconds")
    print("Total function evaluations:", function_evaluations)
    




if __name__ == "__main__":
    x_values = [4,5,6,7,8,9,10]
    for i in range(len(x_values)):
        main(x_values[i])
