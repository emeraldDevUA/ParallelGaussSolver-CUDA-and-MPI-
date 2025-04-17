#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include "functions.h"
#include "MatrixUtils/Matrix.h"

int N = 3;
int threadsPerBlock = 256;





__global__ void eliminate_below(float *A, int n, int pivot) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && row > pivot) {
        float factor = A[row * (n + 1) + pivot];
        for (int col = 0; col <= n; col++) {
            A[row * (n + 1) + col] -= factor * A[pivot * (n + 1) + col];
        }
    }
}
__global__ void back_substitution(float *A, float *x, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        int i = n - row - 1;  // Start from last row
        float sum = 0.0;

        for (int j = i + 1; j < n; j++) {
            sum += A[i * (n + 1) + j] * x[j];
        }

        x[i] = (A[i * (n + 1) + n] - sum) / A[i * (n + 1) + i];
    }
}

__global__ void eliminate_row(double *A, int n, int pivot) {
     __shared__ double pivotRow[6000];
    int row = static_cast<int> (blockIdx.x * blockDim.x + threadIdx.x);

    if (threadIdx.x == 0) {
        for (int col = 0; col <= n; ++col) {
            pivotRow[col] = A[pivot * (n + 1) + col];
        }
    }
    __syncthreads();

    if (row < n && row != pivot) {
        double factor = A[row * (n + 1) + pivot];
        for (int col = 0; col <= n; col++) {
            A[row * (n + 1) + col] -= factor * pivotRow[col];
        }
    }
}
__global__ void normalize_row(double *A, int n, int pivot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double pivot_value;
    if (threadIdx.x == 0) {
        pivot_value = A[pivot * (n + 1) + pivot];
    }
    __syncthreads();

    if (idx < n + 1) {
        A[pivot * (n + 1) + idx] /= pivot_value;  // Normalize pivot row
    }
}



void gauss_jordan_elimination(double *A, int n) {
    double *d_A;
    size_t size = n * (n + 1) * sizeof(double);

    // Allocate GPU memory and copy input matrix
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    for (int pivot = 0; pivot < n; pivot++) {
        // Normalize the pivot row
        int numberOfRuns = n + threadsPerBlock - 1;
        normalize_row<<<numberOfRuns / threadsPerBlock, threadsPerBlock>>>(d_A, n, pivot);
        cudaDeviceSynchronize();

        // Eliminate all other rows (both forward and backward)
        eliminate_row<<<numberOfRuns / threadsPerBlock, threadsPerBlock>>>(d_A, n, pivot);
        cudaDeviceSynchronize();
    }


    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}
//float* gaussElimination(float *A, int n){
//    float*d_A, *d_x;
//    int size_x = N * sizeof(float);
//    size_t size = n*(n+1) * sizeof(float);
//    cudaMalloc((void **)&d_A, size);
//    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
//    cudaMalloc((void**)&d_x, size_x);
//
//    for (int pivot = 0; pivot < n; pivot++) {
//        // Normalize the pivot row
//        normalize_row<<<(n + 255) / 256, 256>>>(d_A, n, pivot);
//        cudaDeviceSynchronize();
//
//        // Eliminate all other rows (both forward and backward)
//        eliminate_below<<<(n + 255) / 256, 256>>>(d_A, n, pivot);
//        cudaDeviceSynchronize();
//    }
//
//    back_substitution<<<1, N>>>(d_A, d_x, N);
//    cudaDeviceSynchronize();
//
//    // Copy result back to host
//    float *h_x = (float*)malloc(sizeof(float)*N);
//    cudaMemcpy(h_x, d_x, size_x, cudaMemcpyDeviceToHost);
//
//    // Print solution
//    std::cout << "Solution:" << std::endl;
//    for (int i = 0; i < N; i++) {
//        std::cout << "x[" << i << "] = " << h_x[i] << std::endl;
//    }
//    // Copy result back to CPU
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N + 1; j++) {
//            printf("%8.3f ", d_A[i*N +j]);
//        }
//        printf("\n");
//    }
//    cudaFree(d_A);
//
//    return h_x;
//}

using namespace chrono;

#define verbose


//#ifdef verbose
//    #undef verbose
//#endif

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cout << "Usage: ./your_program <MatrixSize> <ThreadsPerBlock>" << std::endl;
        return 1;
    }

    if (!parseInt(argv[1], N) || !parseInt(argv[2], threadsPerBlock)) {
        std::cerr << "Invalid input: expected two integers." << std::endl;
        return 1;
    }

    double *matrix1D;

    Matrix<double> matrix1 =
            Matrix<double>(N, N+1, 1.0);

    matrix1.fillRand(10.0);
    matrix1D = matrix1.toArray();



    gauss_jordan_elimination(matrix1D, N);
    cudaDeviceSynchronize();

    auto* xValues = (double*)malloc(sizeof(double) * N);

    #ifdef verbose
       cout << "Roots:" << endl;
       for (int i = 0; i < N; i++) {
        double root = matrix1D[i * (N + 1) + N];
        xValues[i] = root;
        cout << "x[" << i+1 <<"] = " << root <<endl;
    }
    #else
        for (int i = 0; i < N; i++) {
            xValues[i] = matrix1D[i * (N + 1) + N];
        }
    #endif


    #ifdef verbose
        double* matrix1D_copy= matrix1.toArray();
        cout << endl << "Initial Matrix:" << endl;
        printMatrix(matrix1D_copy, N);

        cout << endl<< "Final Matrix:" << endl;
        printMatrix(matrix1D, N);
        cout << endl << "Convergence: "<< endl;
        checkResult(matrix1D_copy, xValues, N);

    free(matrix1D_copy);
    #endif

    free(matrix1D);
    free(xValues);


    return 0;
}



//#include <cuda_runtime.h>
//#include <iostream>
//#include <iomanip>
//
//__global__ void normalize_row(double *A, int n, int pivot) {
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    if (col >= n + 1) return;
//
//    double pivotVal = A[pivot * (n + 1) + pivot];
//    A[pivot * (n + 1) + col] /= pivotVal;
//}
//
//void print_matrix(const double *A, int n, const std::string &label) {
//    std::cout << "\n" << label << ":\n";
//    for (int i = 0; i < n; ++i) {
//        std::cout << "Row " << i << ":\t";
//        for (int j = 0; j <= n; ++j) {
//            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << A[i * (n + 1) + j] << " ";
//        }
//        std::cout << "\n";
//    }
//}
//
//__global__ void eliminate_rows(double *A, int n, int pivot) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    __shared__ double pivotRow[1024]; // Make sure this is large enough for n+1
//
//    // First, all threads help load the pivot row
//    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
//         i < n + 1;
//         i += blockDim.x * blockDim.y) {
//        pivotRow[i] = A[pivot * (n + 1) + i];
//    }
//    __syncthreads();
//
//    // Then perform elimination
//    if (row >= n || col >= n + 1 || row == pivot)
//        return;
//
//    double factor = A[row * (n + 1) + pivot];
//    A[row * (n + 1) + col] -= factor * pivotRow[col];
//}
//
//
//void gauss_jordan_elimination(double *A, int n) {
//    double *d_A;
//    size_t size = n * (n + 1) * sizeof(double);
//
//    cudaMalloc((void **)&d_A, size);
//    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
//
//    dim3 threadsPerBlock(32, 32);
//    dim3 blocksPerGrid(
//            (n + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
//            (n + threadsPerBlock.y - 1) / threadsPerBlock.y
//    );
//
//    for (int pivot = 0; pivot < n; pivot++) {
//        std::cout << "\n=== Pivot Step: " << pivot << " ===";
//
//        // Normalize pivot row
//        int normalizeBlocks = (n + 1 + 255) / 256;
//        normalize_row<<<normalizeBlocks, 256>>>(d_A, n, pivot);
//        cudaDeviceSynchronize();
//
//        cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
//        print_matrix(A, n, "After normalizing pivot row");
//
//        // Eliminate other rows
//        eliminate_rows<<<blocksPerGrid, threadsPerBlock>>>(d_A, n, pivot);
//        cudaDeviceSynchronize();
//
//        cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
//        print_matrix(A, n, "After eliminating other rows");
//    }
//
//    cudaFree(d_A);
//}
//
//int main() {
//    const int n = 3;
//    double A[n][n + 1] = {
//            {2, 1, -1, 8},
//            {-3, -1, 2, -11},
//            {-2, 1, 2, -3}
//    };
//
//    std::cout << "Initial augmented matrix:";
//    print_matrix(&A[0][0], n, "");
//
//    gauss_jordan_elimination(&A[0][0], n);
//
//    std::cout << "\n=== Final Reduced Row Echelon Form ===";
//    print_matrix(&A[0][0], n, "");
//
//    std::cout << "\n=== Solution ===\n";
//    for (int i = 0; i < n; ++i) {
//        std::cout << "x" << i + 1 << " = " << std::fixed << std::setprecision(4) << A[i][n] << "\n";
//    }
//
//    return 0;
//}
