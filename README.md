# CS 239 – Parallel Computing

This repository contains CUDA programming exercises and a final project exploring the application of **metaheuristic optimization algorithms** for curve fitting in **non-linear regression**, implemented in Python and evaluated for both accuracy and runtime performance.

---

## Repository Structure

### CUDA Programming Exercises

#### Exercise 1: Matrix Addition and GPU Profiling
- `Exercise_1_kernel.cu`
- `SALAS_CS 239 EXERCISE 1.pdf`

**Topics Covered:**
- CUDA basics: thread hierarchy, memory transfers
- Matrix addition using `cudaMemcpy` and kernel configuration
- GPU introspection using `cudaGetDeviceProperties`
- CGMA (Compute-to-Global Memory Access) ratio computation

---

#### Exercise 2: Matrix Multiplication – Global vs Shared Memory
- `Exercise_2_kernel.cu`
- `SALAS_CS 239 EXERCISE 2.pdf`

**Topics Covered:**
- Matrix multiplication using:
  - Global memory (`matmul_rec_glob`)
  - Shared memory tiling (`matmul_rec_shar`)
- Execution time analysis using `chrono`
- CGMA computation and performance tuning

---

### Parallel Metaheuristic Algorithms for Hyperparameter Search in Logistic Regression

#### 📄 `SALAS_CS239_FINAL_PROJECT_PAPER.pdf`

**Objective:**  
Use parallel population-based and local search metaheuristic algorithms to find optimal hyperparameters for small machine learning models like **logistic reggression** vs. traditional seuential brute-force methods comparing their optimization effectiveness and computational efficiency.

#### Implemented Algorithms:
| Algorithm                      | File                |
|-------------------------------|---------------------|
| Genetic Algorithm (GA)        | `GA_LR_MSR.py`      |
| Improved GA                   | `GA_LR_MSR2.py`     |
| Simulated Annealing (SA)      | `SA_LR.py`          |
| Tabu Search (TS)              | `TS_LR.py`          |
| Ant Colony Optimization (ACO) | `ACO_LR.py`         |
| Particle Swarm Optimization   | `PSO_LR.py`         |
| Variable Neighborhood Search  | `VNS_LR.py`         |
| Baseline (Sequential)         | `SEQ_LR.py`         |

#### 🧾 Objective:
- Optimize Logistic Regression's Hyperparameters: Regularization parameter C, Penalty mode, Learning rate, and Solver type through parallel metaheuristics 
- Evaluation metrics: **Mean Squared Error (MSE)** and **computation time**
- Tools: Python, matplotlib, `numpy`, `random`, `math`, PyCUDA

**Summary of Findings:**
- Parallel metaheuristics effectively found near-optimal hyperparameters.
- GA variants and PSO produced the best balance between error minimization and runtime.
- Parallel metaheuristic solvers are more time efficient than sequential-brute force methods.
- All results logged in `PROJECT_RESULTS.xlsx`.

---

## Author

**Jeryl Salas**  
Master of Engineering in Artificial Intelligence – University of the Philippines Diliman  
GitHub: [@zencordia4913](https://github.com/zencordia4913)

---

📎 Full reports:
- [Exercise 1 Report](./SALAS_CS%20239%20EXERCISE%201.pdf)
- [Exercise 2 Report](./SALAS_CS%20239%20EXERCISE%202.pdf)
- [Final Project Paper](./SALAS_CS239_FINAL_PROJECT_PAPER.pdf)

> This repo serves as a foundation for metaheuristic research and CUDA learning.
