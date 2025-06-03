#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono>
#include <iomanip>

void queryDeviceProperties() {
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaError_t error;

    error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "We weren't able to detect any device: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    for (int i = 0; i < deviceCount; ++i) {
        error = cudaGetDeviceProperties(&deviceProp, i);
        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << i << ": " << cudaGetErrorString(error) << std::endl;
            continue;
        }
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;

        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    }
}

// Kernel functions
// Kernel function where each thread produces one output matrix element
__global__ void kernel_1t1e(float* A, const float* B, const float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        int idx = i * N + j;
        A[idx] = B[idx] + C[idx];
    }
}

// Kernel function where each thread produces one output matrix row
__global__ void kernel_1t1r(float* A, const float* B, const float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int j = 0; j < N; ++j) {
            int idx = row * N + j;
            A[idx] = B[idx] + C[idx];
        }
    }
}

// Kernel function where each thread produces one output matrix column
__global__ void kernel_1t1c(float* A, const float* B, const float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int i = 0; i < N; ++i) {
            int idx = i * N + col;
            A[idx] = B[idx] + C[idx];
        }
    }
}

void matrixAdditionHost(float* A, const float* B, const float* C, int N, int mode) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (mode == 0) {
        threadsPerBlock.x = deviceProp.maxThreadsPerBlock;
        threadsPerBlock.y = deviceProp.maxThreadsPerBlock;
        numBlocks.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
    }
    else if (mode == 1) {
        threadsPerBlock.x = deviceProp.maxThreadsPerBlock;
        numBlocks.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = 1;
    }
    else if (mode == 2) {
        threadsPerBlock.x = 1;
        numBlocks.x = N;
        numBlocks.y = (N + deviceProp.maxThreadsPerBlock - 1) / deviceProp.maxThreadsPerBlock;
    }
    else {
        std::cerr << "Invalid mode. Using element-wise addition." << std::endl;
        threadsPerBlock.x = deviceProp.maxThreadsPerBlock;
        threadsPerBlock.y = deviceProp.maxThreadsPerBlock;
        numBlocks.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    switch (mode) {
    case 0:
        kernel_1t1e << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
        break;
    case 1:
        kernel_1t1r << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
        break;
    case 2:
        kernel_1t1c << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
        break;
    default:
        std::cerr << "Invalid mode entered. Using element-wise addition instead." << std::endl;
        kernel_1t1e << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
    }

    cudaMemcpy(A, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void printMatrix(const float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void generateRandomMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    }
}

double calculateGFLOPS(int N, double duration) {
    double numOps = 2.0 * N * N;
    double FLOPS = numOps / duration;
    double GFLOPS = FLOPS / 1e9;
    return GFLOPS;
}

int main() {
    // Clear terminal screen
#ifdef _WIN32
    std::system("cls");
#else
    std::system("clear");
#endif

    srand(time(NULL));
    queryDeviceProperties();
    std::cout << "\n";
    std::cout << "--------------------------Device 0 performance--------------------------" << std::endl;
    std::cout << "\n";

    int sizes[] = { 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824 };
    const char* modeDescriptions[] = { "     Element-wise", "        Row-wise", "        Column-wise" };

    // Print header of the runtime table
    std::cout << std::setw(15) << "Matrix Size";
    for (const auto& modeDesc : modeDescriptions) {
        std::cout << std::setw(20) << modeDesc;
    }
    std::cout << std::endl;

    // Print runtime table
    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int N = sizes[i];

        std::cout << std::setw(16) << N;

        for (int mode = 0; mode <= 2; ++mode) {
            auto start = std::chrono::high_resolution_clock::now();

            float* B = new float[N * N];
            float* C = new float[N * N];
            float* A = new float[N * N];

            generateRandomMatrix(B, N);
            generateRandomMatrix(C, N);

            matrixAdditionHost(A, B, C, N, mode);

            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;

            std::cout << std::setw(20) << duration.count() << " s";

            delete[] B;
            delete[] C;
            delete[] A;
        }
        std::cout << std::endl;
    }

    std::cout << "\n";

    // Print header of the GFLOPS table
    std::cout << std::setw(15) << "Matrix Size";
    for (const auto& modeDesc : modeDescriptions) {
        std::cout << std::setw(20) << modeDesc;
    }
    std::cout << std::endl;

    // Print GFLOPS table
    for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int N = sizes[i];

        std::cout << std::setw(16) << N;

        for (int mode = 0; mode <= 2; ++mode) {
            auto start = std::chrono::high_resolution_clock::now();

            float* B = new float[N * N];
            float* C = new float[N * N];
            float* A = new float[N * N];

            generateRandomMatrix(B, N);
            generateRandomMatrix(C, N);

            matrixAdditionHost(A, B, C, N, mode);

            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;

            double GFLOPS = calculateGFLOPS(N, duration.count());
            std::cout << std::setw(20) << GFLOPS << " GFLOPS";

            delete[] B;
            delete[] C;
            delete[] A;
        }
        std::cout << std::endl;
    }

    return 0;
}
