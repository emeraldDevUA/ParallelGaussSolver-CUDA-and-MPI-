
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <algorithm>


#include "functions.h"
#include "../MatrixUtils/Matrix.h"

//#define E 1e-6

using namespace std;




#define GAUSS_DEFAULT 0
#define GAUSS_JORDAN  1



#define verbose

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            cout << "Usage: " << argv[0] << " <matrix_size> <0/1>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    const int N = atoi(argv[1]);
    const int type = atoi(argv[2]);


    if (N <= 0) {
        if (rank == 0) {
            cout << "Matrix size must be positive!" << endl;
        }
        MPI_Finalize();
        return -1;
    }



    const int base_rows = N / size;
    const int extra = N % size;
    const int local_rows = base_rows + (rank < extra ? 1 : 0);
    const int local_size = local_rows * (N + 1);


    double* A = nullptr;
    double* A_copy = nullptr;
    double* local_A = new double[local_size];
    double* pivot_row = new double[N + 1];


    vector<int> recvcounts(size), displs(size);

    // Calculate displacements and counts
    int offset = 0;

    for (int i = 0; i < size; i++) {
        int rows = base_rows + (i < extra ? 1 : 0);
        recvcounts[i] = rows * (N + 1);
        displs[i] = offset;
        offset += recvcounts[i];
    }

    // Generate matrix on root
    if (rank == 0) {
        Matrix<double> matrix1 =
                Matrix<double>(N, N+1, 1.0);

        matrix1.fillRand(10.0);

        A = matrix1.toArray();

        A_copy = matrix1.toArray();

    }

    // Scatter matrix
    MPI_Scatterv(A, recvcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A, local_size, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Timing
    double start_time = MPI_Wtime();

    if (type == GAUSS_DEFAULT) {
        // Gaussian elimination
        for (int k = 0; k < N; k++) {
            // Determine which process owns the current pivot row
            int owner = (k < (base_rows + 1) * extra)
                ? k / (base_rows + 1)
                : extra + (k - (base_rows + 1) * extra) / base_rows;

            // Get pivot row
            if (rank == owner) {
                int local_k = k - displs[owner] / (N + 1);
                copy(local_A + local_k * (N + 1), local_A + (local_k + 1) * (N + 1), pivot_row);
            }
            MPI_Bcast(pivot_row, N + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Elimination
            for (int i = 0; i < local_rows; i++) {
                int global_i = displs[rank] / (N + 1) + i;
                if (global_i > k) {

                    double factor = local_A[i * (N + 1) + k] / pivot_row[k];

                    for (int j = k; j < N + 1; j++) {
                        local_A[i * (N + 1) + j] -= factor * pivot_row[j];
                    }
                }

            }
        }

        // Gather results
        MPI_Gatherv(local_A, local_size, MPI_DOUBLE,
            A, recvcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);


        if (rank == 0) {
            double* x = new double[N];




            // Timing
            double elim_time = MPI_Wtime() - start_time;
            printf("Elimination time: %.4f seconds\n", elim_time);

            start_time = MPI_Wtime();

            for (int i = N - 1; i >= 0; i--) {
                x[i] = A[i * (N + 1) + N];
                for (int j = i + 1; j < N; j++) {
                    x[i] -= A[i * (N + 1) + j] * x[j];
                }
                x[i] /= A[i * (N + 1) + i];
            }

            double backsub_time = MPI_Wtime() - start_time;
            printf("Back substitution time: %.4f seconds\n", backsub_time);
            printf("Total time: %.4f seconds\n", elim_time + backsub_time);

            // Print solution (only for small matrices)
#ifdef verbose
            cout << "Roots:" << endl;
            for (int i = 0; i < N; i++) {
                printf("x[%d] = %.6f\n", i, x[i]);
            }

            cout << endl << "Initial Matrix:" << endl;
            printMatrix(A_copy, N);
            cout << endl << "Final Matrix:" << endl;
            printMatrix(A, N);


            cout << endl << "Convergence: "<< endl;
            checkResult(A_copy, x, N);
            free(A_copy);
#endif
            delete[] x;
        }

        // Clean up
        if (rank == 0) {
            delete[] A;
        }
        delete[] local_A;
        delete[] pivot_row;
    }
    else if (type == GAUSS_JORDAN) {
        for (int k = 0; k < N; k++) {
            // Determine which process owns the current pivot row
            int owner = (k < (base_rows + 1) * extra)
                ? k / (base_rows + 1)
                : extra + (k - (base_rows + 1) * extra) / base_rows;

            // Get pivot row and normalize
            if (rank == owner) {
                int local_k = k - displs[owner] / (N + 1);
                double pivot_val = local_A[local_k * (N + 1) + k];

                for (int j = 0; j < N + 1; j++) {
                    pivot_row[j] = local_A[local_k * (N + 1) + j] / pivot_val;
                }

                // Write normalized row back
                for (int j = 0; j < N + 1; j++) {
                    local_A[local_k * (N + 1) + j] = pivot_row[j];
                }
            }

            // Broadcast normalized pivot row
            MPI_Bcast(pivot_row, N + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Eliminate current column from all other rows
            for (int i = 0; i < local_rows; i++) {
                int global_i = displs[rank] / (N + 1) + i;
                if (global_i != k) {
                    double factor = local_A[i * (N + 1) + k];
                    for (int j = 0; j < N + 1; j++) {
                        local_A[i * (N + 1) + j] -= factor * pivot_row[j];
                    }
                }
            }
        }

        // Gather final matrix at root
        MPI_Gatherv(local_A, local_size, MPI_DOUBLE,
            A, recvcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // Extract solution vector
        if (rank == 0) {


            std::vector<double> x(N);
            for (int i = 0; i < N; i++) {
                x[i] = A[i * (N + 1) + N]; // Since LHS is now identity matrix
            }


#ifdef verbose
            cout << "Roots:" << endl;
            for (int i = 0; i < N; i++) {
                cout << "x[" << i+1 <<"] = " << x[i] <<endl;
            }

            cout << endl << "Initial Matrix:" << endl;
            printMatrix(A_copy, N);
            cout << endl << "Final Matrix:" << endl;
            printMatrix(A, N);

            cout << endl << "Convergence: "<< endl;
            checkResult(A_copy, (&x[0]), N);
            free(A_copy);
#endif
        }

    }


    MPI_Finalize();
    return 0;
}
