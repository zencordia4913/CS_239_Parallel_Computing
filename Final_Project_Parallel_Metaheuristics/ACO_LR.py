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

# CUDA kernel code for updating pheromone levels
update_pheromone_kernel = """
__global__ void evaluate_fitness(float *hyperparameters, float *data, float *labels, float *accuracies,
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


__global__ void update_pheromone(float *pheromone, int *best_hyperparameters, int num_penalty_mode, int num_C_values, int num_Solver, int num_Learning_rate, float evaporation_rate) {
    int penalty_mode_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (penalty_mode_idx < num_penalty_mode) {
        for (int C_values_idx = 0; C_values_idx < num_C_values; ++C_values_idx) {
            for (int Solver_idx = 0; Solver_idx < num_Solver; ++Solver_idx) {
                for (int Learning_rate_idx = 0; Learning_rate_idx < num_Learning_rate; ++Learning_rate_idx) {
                    int idx = (penalty_mode_idx * num_C_values * num_Solver * num_Learning_rate) + (C_values_idx * num_Solver * num_Learning_rate) + (Solver_idx * num_Learning_rate) + Learning_rate_idx;
                    if (penalty_mode_idx == best_hyperparameters[0] && C_values_idx == best_hyperparameters[1] && Solver_idx == best_hyperparameters[2] && Learning_rate_idx == best_hyperparameters[3]) {
                        pheromone[idx] *= (1.0 - evaporation_rate);
                        pheromone[idx] += evaporation_rate;
                    } else {
                        pheromone[idx] *= (1.0 - evaporation_rate);
                    }
                }
            }
        }
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


# Define the Ant Colony Optimization algorithm for hyperparameter search using PyCUDA
class AntColonyOptimizationCUDA:
    def __init__(self, num_ants, num_iterations, num_penalty_mode, num_C_values, num_Solver, num_Learning_rate, pheromone_init, alpha, beta, evaporation_rate):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_penalty_mode = num_penalty_mode
        self.num_C_values = num_C_values
        self.num_Solver = num_Solver
        self.num_Learning_rate = num_Learning_rate
        self.pheromone = np.ones((num_penalty_mode, num_C_values, num_Solver, num_Learning_rate), dtype=np.float32) * pheromone_init
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.best_hyperparameters = None
        self.best_accuracy = -1
        self.cpu_time = None
        self.gpu_time = None
        self.function_evaluation = None

        # Determine optimal block size and grid size
        device_attributes = drv.Device(0).get_attributes()
        max_threads_per_block = device_attributes[3]  # Max threads per block
        self.block_size = min(256, max_threads_per_block)  # Adjust the block size according to device properties
        self.grid_size = (num_penalty_mode + self.block_size - 1) // self.block_size

        # Compile CUDA kernel
        self.mod = SourceModule(update_pheromone_kernel)
        self.update_pheromone_cuda = self.mod.get_function("update_pheromone")
        self.evaluate_fitness = self.mod.get_function("evaluate_fitness")

    def select_hyperparameters(self):
        probabilities = self.pheromone ** self.alpha  # pheromone-based selection
        heuristic_info = np.random.rand(self.num_penalty_mode, self.num_C_values, self.num_Solver, self.num_Learning_rate) ** self.beta  # random exploration
        combined_probs = probabilities * heuristic_info
        total_probs = np.sum(combined_probs)
        if total_probs == 0:
            combined_probs = np.ones_like(combined_probs)  # if all probabilities are zero, use uniform distribution
            total_probs = np.sum(combined_probs)
        normalized_probs = combined_probs / total_probs
        selected_indices = np.unravel_index(np.argmax(normalized_probs), normalized_probs.shape)
        return selected_indices

    def update_pheromone(self):
        best_hyperparameters = self.best_hyperparameters
        evaporation_rate = np.float32(self.evaporation_rate)

        self.update_pheromone_cuda(drv.Out(self.pheromone), drv.In(np.array(best_hyperparameters, dtype=np.int32)),
                                    np.int32(self.num_penalty_mode), np.int32(self.num_C_values), np.int32(self.num_Solver), 
                                    np.int32(self.num_Learning_rate), evaporation_rate,
                                    block=(self.block_size, 1, 1), grid=(self.grid_size, 1))

    def search(self, X_train_scaled, X_train, y_train, ants, accuracies):
        # Metrics
        start_time = time.time()
        gpu_start = drv.Event()
        gpu_end = drv.Event()
        gpu_start.record()
        function_evaluations = 0

        for iteration in range(self.num_iterations):
            ants_results = []
            for ant in range(self.num_ants):
                ants_results.append(self.select_hyperparameters())

            # Allocate memory on the device
            X_train_gpu = drv.mem_alloc(X_train_scaled.nbytes)
            y_train_gpu = drv.mem_alloc(y_train.nbytes)
            hyperparameters_gpu = drv.mem_alloc(ants.nbytes)
            accuracies_gpu = drv.mem_alloc(accuracies.nbytes)

            # Transfer data to the device
            drv.memcpy_htod(X_train_gpu, X_train_scaled)
            drv.memcpy_htod(y_train_gpu, y_train)
            drv.memcpy_htod(hyperparameters_gpu, ants)
            
            num_features = X_train.shape[1]
            num_samples = X_train.shape[0]
            num_ants = ants.shape[0]
            
            # Define grid and block dimensions
            block_size = 256
            grid_size = (num_ants + block_size - 1) // block_size

            # Calculate shared memory size
            shared_memory_size = (2 * num_features) * 4  # 2 arrays of floats of size num_features
            
            # Launch the kernel
            try:
                self.evaluate_fitness(hyperparameters_gpu, X_train_gpu, y_train_gpu, accuracies_gpu,
                                    np.int32(num_features), np.int32(num_samples), np.int32(num_ants),
                                    block=(block_size, 1, 1), grid=(grid_size, 1, 1), shared=shared_memory_size)
                function_evaluations += num_ants
            except drv.driver.CudaAPIError as e:
                print("CUDA error:", e)
                
            drv.memcpy_dtoh(accuracies, accuracies_gpu)
            best_index = np.argmax(accuracies)
            best_hyperparameters = ants[best_index]
            
            self.best_hyperparameters = best_hyperparameters
            self.best_accuracy = accuracies[best_index]
            
            # Print progress
            #print("\n")    
            #print(f"Iteration: {iteration}, Best Fitness: {self.best_accuracy}")
            #print("\n")    

            self.update_pheromone()
        gpu_end.record()
        gpu_end.synchronize()

        end_time = time.time()

        cpu_time = end_time - start_time
        gpu_time = gpu_start.time_till(gpu_end) * 1e-3
        self.cpu_time = cpu_time
        self.gpu_time = gpu_time
        self.function_evaluation = function_evaluations


# Load the cancer dataset
process = psutil.Process()
memory_start = process.memory_info().rss  

x_values = [1,2,3,4,5,6,7,8,9,10]

for i in range(len(x_values)):
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

    # Hyperparameters for ACO
    num_ants = 200
    num_iterations=50
    pheromone_init=0.01
    alpha=1.0
    beta=2.0
    evaporation_rate=0.8

    # Initialize population randomly
    ants = np.empty((num_ants, hyperparameter_length), dtype=np.float32)
    for i in range(num_ants):
        ants[i, 0] = np.random.randint(0, len(penalty_mod))  # Randomly select penalty index
        ants[i, 1] = C_values[np.random.randint(0, len(C_values))]  # Randomly select regularization constant
        ants[i, 2] = np.random.randint(0, len(Solver))  # Randomly select solver index
        ants[i, 3] = Learning_rates[np.random.randint(0, len(Learning_rates))]  # Randomly select learning rate
    
    accuracies = np.zeros(num_ants, dtype=np.float32)
    X_train_scaled = X_train_scaled.flatten().astype(np.float32)

    # Start time
    start_time = time.time()

    # Initialize Ant Colony Optimization algorithm using PyCUDA
    aco_cuda = AntColonyOptimizationCUDA(num_ants, num_iterations, num_penalty_mod, num_C_values, 
                                         num_Solver, num_Learning_rates, pheromone_init, alpha, beta, evaporation_rate)

    # Perform hyperparameter search
    aco_cuda.search(X_train_scaled, X_train, y_train, ants, accuracies)

    # End time
    end_time = time.time()

    # Calculate duration
    duration = end_time - start_time

    # Retrieve best hyperparameters and accuracy
    best_hyperparameters = aco_cuda.best_hyperparameters
    best_accuracy = aco_cuda.best_accuracy
    cpu_time = aco_cuda.cpu_time
    gpu_time = aco_cuda.gpu_time
    function_evaluations = aco_cuda.function_evaluation

    # Find the best solution 
    print("\n")
    print("\n")
    print("---------Hyperparameter Search Summary----------")
    print("ML Model used: Logistic Regression")
    print("Dataset: Breast cancer dataset")
    print("Hyperparameter search method: Parallel Ant Colony Optimization")
    print("Parallel strategy: 1C/RS/SPSS")
    best_penalty_idx = int(best_hyperparameters[0])
    best_solution_idx = int(best_hyperparameters[2])
    print("\n")
    print("-----Best hyperparameter combination found-----")
    print("Penalty: ", penalty_mod_mapping[best_penalty_idx])
    print("Regularization constant, C: ", best_hyperparameters[1])
    print("Solution: ", Solver_mapping[best_solution_idx])
    print("Learning rate: ", best_hyperparameters[3])
    print("Best fitness (accuracy):", best_accuracy)
    print("Total search time:", duration, "seconds")
    print("\n")
    print("-----Evaluating best hyperparameters on test set-----")
    accuracy, precision, recall, f1, mcc, confusion_mat = evaluate_test_set(X_test, y_test, X_train, y_train, best_hyperparameters, penalty_mod_mapping, Solver_mapping)
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

    
