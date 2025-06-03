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

# CUDA kernel code for evaluating neighbors
cuda_code = """
__global__ void evaluate_neighbors_kernel(float *hyperparameters, float *data, float *labels, float *accuracies,
                                 int num_features, int num_samples, int pop_size, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < pop_size) {
        int penalty_idx = int(hyperparameters[4 * idx]);
        float C = hyperparameters[4 * idx + 1];
        int solver_idx = int(hyperparameters[4 * idx + 2]); 
        float l_r = hyperparameters[4 * idx + 3];
        
        // Initialize hyperparameters
        float penalty = penalty_idx == 0 ? 0 : 1; // Assuming 0 for L1 and 1 for L2

        // Allocate global memory for weights and gradients
        float *weight = new float[num_features];
        float *gradients = new float[num_features];
        float initial_weight = idx * 0.01;

        // Initialize weights with small random values
        for (int i = 0; i < num_features; i++) {
            weight[i] = initial_weight;
        }

        float learning_rate = l_r;
        int num_iterations = 100; // Increase number of iterations

        for (int iter = 0; iter < num_iterations; iter++) {
            // Initialize gradients to zero
            for (int j = 0; j < num_features; j++) {
                gradients[j] = 0.0;
            }
            // Compute gradients
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

            // Update weights with L1 or L2 penalty based on the value of penalty
            if (penalty == 0) { // L1 penalty
                for (int j = 0; j < num_features; j++) {
                    float regularization_term = (weight[j] > 0) ? C : -C;
                    weight[j] -= learning_rate * (gradients[j] / num_samples + regularization_term / num_samples);
                }
            } else if (penalty == 1) { // L2 penalty
                for (int j = 0; j < num_features; j++) {
                    weight[j] -= learning_rate * (gradients[j] / num_samples + 2 * C * weight[j]);
                }
            }
        }

        // Calculate accuracy
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

        // Print debug messages
        //printf("Thread %d: penalty_idx=%d, C_idx=%f, solver_idx=%d, accuracy=%f\\n",
               //idx, penalty_idx, C, solver_idx, accuracies[idx]);

        // Free dynamically allocated memory
        delete[] weight;
        delete[] gradients;
    }
}
"""

def evaluate_test_set(X_test, y_test, X_train, y_train, best_hyperparameters, penalty_mod_mapping, Solver_mapping):
    # Extract hyperparameters
    penalty_idx = int(best_hyperparameters[0])
    C = best_hyperparameters[1]
    solver_idx = int(best_hyperparameters[2])
    learning_rate = best_hyperparameters[3]

    # Initialize logistic regression model with best hyperparameters
    model = LogisticRegression(penalty=penalty_mod_mapping[penalty_idx],
                               C=C,
                               solver=Solver_mapping[solver_idx],
                               max_iter=1000)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, mcc, confusion_mat



# Define the Simulated Annealing algorithm
def simulated_annealing(num_iterations, num_neighbors, num_penalty_mode, num_C_values, num_Solver, num_Learning_rates, initial_temperature, cooling_rate, X_train_scaled, X_train, y_train):
    mod = SourceModule(cuda_code)
    evaluate_neighbors_cuda = mod.get_function("evaluate_neighbors_kernel")
    
    best_solution = None
    best_energy = float('-inf')
    neighborhood_size = 2
    prev_energies = np.zeros(num_neighbors, dtype=np.float32)
    
    # Define the function to generate hyperparameters
    def generate_hyperparameters(current_solution, num_neighbors, a, Learning_rates, C_values):
        neighbors = []
        for _ in range(num_neighbors):
            p = np.clip(current_solution[0] + np.random.randint(-a, a+1), 0, num_penalty_mod - 1)
            c = np.clip(current_solution[1] + np.random.randint(-a, a+1), C_values.min(), C_values.max())
            s = np.clip(current_solution[2] + np.random.randint(-a, a+1), 0, num_Solver - 1)
            l = np.clip(current_solution[3] + np.random.randint(-a, a+1), Learning_rates.min(), Learning_rates.max())
            neighbors.append((p, c, s, l))
        return np.array(neighbors, dtype=np.float32) 
    
    # Simulated annealing main loop
    current_solution = generate_hyperparameters((0, 0, 0, 0), 1, neighborhood_size, Learning_rates, C_values)[0]  # Start with a random solution
    current_temperature = initial_temperature
    num_features = X_train.shape[1]
    num_samples = X_train.shape[0]
    
    # Metrics
    start_time = time.time()
    gpu_start = drv.Event()
    gpu_end = drv.Event()
    gpu_start.record()
    function_evaluations = 0
    
    for i in range(num_iterations):
        # Generate neighbors
        neighbors = generate_hyperparameters(current_solution, num_neighbors, neighborhood_size, Learning_rates, C_values)
        d_neighbors = drv.mem_alloc(neighbors.nbytes)
        drv.memcpy_htod(d_neighbors, neighbors)
        
        # Allocate memory on GPU for results
        energies = np.empty(len(neighbors), dtype=np.float32)
        d_energies = drv.mem_alloc(len(neighbors) * np.float32().itemsize)
        
        # Allocate memory on the device
        X_train_gpu = drv.mem_alloc(X_train_scaled.nbytes)
        y_train_gpu = drv.mem_alloc(y_train.nbytes)
        hyperparameters_gpu = drv.mem_alloc(neighbors.nbytes)

        # Transfer data to the device
        drv.memcpy_htod(X_train_gpu, X_train_scaled)
        drv.memcpy_htod(y_train_gpu, y_train)
        drv.memcpy_htod(hyperparameters_gpu, neighbors)
    
        # Define grid and block dimensions
        block_size = 256
        grid_size = (num_neighbors + block_size - 1) // block_size

        # Calculate shared memory size
        shared_memory_size = (2 * num_features) * 4  # 2 arrays of floats of size num_features

        # Evaluate fitness values of neighbors using CUDA
        evaluate_neighbors_cuda(hyperparameters_gpu, X_train_gpu, y_train_gpu, d_energies,
                             np.int32(num_features), np.int32(num_samples), np.int32(num_neighbors),
                             block=(block_size, 1, 1), grid=(grid_size, 1, 1), shared=shared_memory_size)
        function_evaluations += num_neighbors

        # Copy fitness values from device to host
        drv.memcpy_dtoh(energies, d_energies)
        
        # Compute change in energy
        delta_energy = energies - prev_energies

        # Acceptance probability or Metropolis criterion
        acceptance_probabilities = np.exp(-delta_energy / current_temperature)

        # Randomly accept or reject the proposed solution
        accept_indices = np.where(np.random.rand(len(neighbors)) < acceptance_probabilities)[0]
        for index in accept_indices:
            current_solution = neighbors[index]

        # Update best solution and energy if necessary
        if np.max(energies) > best_energy:
            best_energy = np.max(energies)
            best_solution = current_solution
        
        #print("\n")
        #print(f'After {i} iterations, Best solution: {best_solution}, Best energy: {best_energy}')
        #print("\n")
        
        # Cool down the temperature
        current_temperature *= cooling_rate
        prev_energies = energies
        
    gpu_end.record()
    gpu_end.synchronize()

    end_time = time.time()

    cpu_time = end_time - start_time
    gpu_time = gpu_start.time_till(gpu_end) * 1e-3
        
    return best_solution, best_energy, gpu_time, cpu_time, function_evaluations


if __name__ == "__main__":
    x_values = [1,2,3,4,5,6,7,8,9,10]
    for i in range(len(x_values)):
        process = psutil.Process()
        memory_start = process.memory_info().rss
    
        # Load the cancer dataset
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        #print(X_train_scaled)
        X_train_scaled = X_train_scaled.flatten().astype(np.float32)

        # Hyperparameter search space
        hyperparameter_length = 4
        penalty_mod_mapping = ["l1", "l2"]
        penalty_mod = [0, 1]
        C_values = np.logspace(-1, 1, num=10*x_values[i])
        Solver_mapping = ["liblinear", "saga"]
        Solver = [0, 1]
        Learning_rates = np.logspace(-1, 0, num=10*x_values[i])

        num_penalty_mod = len(penalty_mod)
        num_C_values = len(C_values)
        num_Solver = len(Solver)
        num_Learning_rates = len(Learning_rates)
    
        # Define parameters of Simulated Annealing
        num_iterations = 100
        num_neighbors = 30
        initial_temperature = 100.0
        cooling_rate = 0.95
    
        # Start time
        start_time = time.time()
    
        best_solution, best_fitness, gpu_time, cpu_time, function_evaluations = simulated_annealing(num_iterations, num_neighbors, num_penalty_mod, num_C_values, 
                                                          num_Solver, num_Learning_rates, initial_temperature, cooling_rate, 
                                                          X_train_scaled, X_train, y_train)
    
        # End time
        end_time = time.time()

        # Calculate duration
        duration = end_time - start_time
    
        # Find the best solution 
        print("\n")
        print("\n")
        print("---------Hyperparameter Search Summary----------")
        print("ML Model used: Logistic Regression")
        print("Dataset: Breast cancer dataset")
        print("Hyperparameter search method: Parallel Simulated Annealing")
        print("Parallel strategy: 1C/RS/SPSS")
        best_penalty_idx = int(best_solution[0])
        best_solution_idx = int(best_solution[2])
        print("\n")
        print("-----Best hyperparameter combination found-----")
        print("Penalty: ", penalty_mod_mapping[best_penalty_idx])
        print("Regularization constant, C: ", best_solution[1])
        print("Solution: ", Solver_mapping[best_solution_idx])
        print("Learning rate: ", best_solution[3])
        print("Best fitness (accuracy):", best_fitness)
        print("Total search time:", duration, "seconds")
        print("\n")
        print("-----Evaluating best hyperparameters on test set-----")
        accuracy, precision, recall, f1, mcc, confusion_mat = evaluate_test_set(X_test, y_test, X_train, y_train, best_solution, penalty_mod_mapping, Solver_mapping)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("MCC:", mcc)
        print("Confusion Matrix:\n", confusion_mat)
        print("-----Parallel Search Strategy Performance-----")
        memory_end = process.memory_info().rss
        memory_usage = (memory_end - memory_start) / (1024 * 1024)
        print("Memory usage:", memory_usage, "MB")
        print("Total search time (CPU):", cpu_time, "seconds")
        print("Total search time (GPU):", gpu_time, "seconds")
        print("Total function evaluations:", function_evaluations)
        print("\n")
        print("\n")
        print("\n")
