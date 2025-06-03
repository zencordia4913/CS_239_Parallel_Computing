#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono> // For measuring runtime
#include <iomanip> // For displaying results on a table

#define TILE_SIZE 32

// We want to print the device properties
void queryGPUProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device Name: " << deviceProp.name << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
        // Added more a bit of device properties compared to the first exercise (Baed on Reference 3)
    }
}

__global__ void matmul_rec_glob(float* A, float* B, float* C, int n, int k, int m, float* num_operations, float* num_globmem_acc) {
    // Matrix multiplication function using global memory (based on page 67)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        float sum = 0.0f;
        float operations = 0.0f;
        float accesses = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * m + col];
            operations += 2.0f; // One multiplication + addition and therefore +2 operations per iter. This is my own code
            accesses += 2.0f; // Accessing A[i] and B[i] per iter and thus +2 accesses per iter
        }
        C[row * m + col] = sum;
        num_operations[row * m + col] = operations; // store number of operations and accesses on the matrices. This is my own code
        num_globmem_acc[row * m + col] = accesses;
    }
}

__global__ void matmul_rec_shar(float* A, float* B, float* C, int n, int k, int m, float* num_operations, float* num_globmem_acc) {
    // Matrix multiplication function using shared memory (based on page 87)
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0f;
    float operations = 0.0f;
    float accesses = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (Row < n && t * TILE_SIZE + tx < k) {
            s_A[ty][tx] = A[Row * k + t * TILE_SIZE + tx];
        }
        else {
            s_A[ty][tx] = 0.0f;
        }

        if (Col < m && t * TILE_SIZE + ty < k) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * m + Col];
        }
        else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            Cvalue += s_A[ty][i] * s_B[i][tx];
            operations += 2.0f; // One multiplication + addition and therefore +2 operations per iter. This is my own code
            accesses += 2.0f; // Accessing A[i] and B[i] per iter and thus +2 accesses per iter
        }

        __syncthreads();
    }

    if (Row < n && Col < m) {
        C[Row * m + Col] = Cvalue;
        num_operations[Row * m + Col] = operations; // store number of operations and accesses on the matrices. This is my own code
        num_globmem_acc[Row * m + Col] = accesses;
    }
}


// We want to generate random matrices
void generateRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    }
}

// We want a function that computes the CGMA ratio. My own code
float computeCGMARatio(float* num_operations, float* num_globmem_acc, int n, int m) {
    // We calculate total CGMA ratio
    float total_cgma_ratio = 0.0f;
    for (int j = 0; j < n * m; ++j) {
        if (num_globmem_acc[j] != 0) {
            total_cgma_ratio += num_operations[j] / num_globmem_acc[j];
        }
    }

    // We calculate average CGMA ratio for all elements in the matrix
    float average_cgma_ratio = total_cgma_ratio / (n * m);

    return average_cgma_ratio;
}


int main() {
    // Query and print GPU properties
    queryGPUProperties();

    std::cout << "\n";
    std::cout << "--------------------------Device 0 performance--------------------------" << std::endl;
    std::cout << "\n";

    std::cout << std::setw(5) << "Matrix Size" << std::setw(25) << "Global Memory Runtime(sec)" << std::setw(25) << "Shared Memory Runtime(sec)" << std::setw(25) << "Global Memory CGMA ratio" << std::setw(25) << "Shared Memory CGMA ratio" << std::endl;
    std::cout << std::string(140, '-') << std::endl;

    // Loop through different matrix sizes n, k, m = 32*i where i is the iteration number
    for (int i = 1; i <= 2; ++i) {
        // Initializing matrices and matrix row and column sizes
        float* A, * B, * C;
        int n = 32 * i;
        int k = 32 * i;
        int m = 32 * i;
        float CGMA_ratio_glob; // I call these two "Metric matrices" used to compute CGMA 
        float CGMA_ratio_shar;

        size_t size_A = n * k * sizeof(float);
        size_t size_B = k * m * sizeof(float);
        size_t size_C = n * m * sizeof(float);
        cudaMallocManaged(&A, size_A);
        cudaMallocManaged(&B, size_B);
        cudaMallocManaged(&C, size_C);

        // Allocating memory for matrices and setting "Metric matrices" to a matrix full of zeroes. These matrices will be filled soon
        float* num_operations, * num_globmem_acc;
        cudaMallocManaged(&num_operations, size_C * sizeof(float));
        cudaMallocManaged(&num_globmem_acc, size_C * sizeof(float));
        cudaMemset(num_operations, 0, size_C * sizeof(float));
        cudaMemset(num_globmem_acc, 0, size_C * sizeof(float));

        // We generate random matrices A and B
        generateRandomMatrix(A, n, k);
        generateRandomMatrix(B, k, m);

        // We define block and grid dimensions by the TILE_SIZE that is set
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        // We perform matrix multiplication using global memory and find the runtime duration
        auto start = std::chrono::high_resolution_clock::now();
        matmul_rec_shar << <gridSize, blockSize >> > (A, B, C, n, k, m, num_operations, num_globmem_acc);
        cudaDeviceSynchronize(); 
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        CGMA_ratio_glob = computeCGMARatio(num_operations, num_globmem_acc, n, m); // We compute the global memory's CGMA ratio. My own code
        cudaMemset(num_operations, 0, size_C * sizeof(float));
        cudaMemset(num_globmem_acc, 0, size_C * sizeof(float));

        // We perform matrix multiplication using shared memory and find the runtime duration
        auto start_2 = std::chrono::high_resolution_clock::now();
        matmul_rec_glob << <gridSize, blockSize >> > (A, B, C, n, k, m, num_operations, num_globmem_acc);
        cudaDeviceSynchronize(); 
        auto end_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_2 = end_2 - start_2;

        CGMA_ratio_shar = computeCGMARatio(num_operations, num_globmem_acc, n, m); // We compute the global memory's CGMA ratio. My own code

        // Print results in a table
        std::cout << std::setw(5) << n << std::setw(25) << duration_2.count() << std::setw(25) << duration.count() << std::setw(25) << CGMA_ratio_glob << std::setw(25) << CGMA_ratio_shar << std::endl;


        // Was adviced to free up memory (page 51)
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    return 0;
}
