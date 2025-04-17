//
// Created by GameRock on 11/04/2025.
//

#ifndef CUDASAMPLE_FUNCTIONS_H
#define CUDASAMPLE_FUNCTIONS_H


#include <climits>

void checkResult(double* augmentedMatrix, double* roots, int n) {
    const double E = 1e-8;  // Convergence threshold
    int numberConverged = 0;
    double deviation = 0.0;

    for (int i = 0; i < n; ++i) {
        double temp = 0.0;

        // Compute left-hand side of the equation
        for (int j = 0; j < n; ++j) {
            temp += augmentedMatrix[i * (n + 1) + j] * roots[j];
        }

        // Calculate deviation from right-hand side
        double local_deviation = std::abs(temp - augmentedMatrix[i * (n + 1) + n]);
        deviation += local_deviation;

        if (local_deviation <= E) {
            std::cout << "Equation #" << (i + 1) << " converges!" << std::endl;
            numberConverged++;
        }
    }

    std::cout << endl << "Number of Converged Equations: " << numberConverged
              << " | Total Deviation: " << deviation << std::endl;
}


void printMatrix(const double* A, int n) {
    const int maxPrint = 6; // control how much of the matrix to show
    int rowsToPrint = std::min(n, maxPrint);
    int colsToPrint = std::min(n + 1, maxPrint);

    for (int i = 0; i < rowsToPrint; ++i) {
        for (int j = 0; j < colsToPrint; ++j) {
            printf("%8.3lf ", A[i * (n + 1) + j]);
        }

        if (colsToPrint < n + 1) {
            printf("... ");
            printf("%8.3lf", A[i * (n + 1) + n]); // last column (b)
        }
        printf("\n");
    }

    if (rowsToPrint < n) {
        printf("...\n");

        // Optionally print the last row (bottom-right corner)
        for (int j = 0; j < colsToPrint; ++j) {
            printf("%8.3lf ", A[(n - 1) * (n + 1) + j]);
        }
        if (colsToPrint < n + 1) {
            printf("... ");
            printf("%8.3lf", A[(n - 1) * (n + 1) + n]);
        }
        printf("\n");
    }
}


bool parseInt(const char* str, int& out) {
    char* endptr;
    errno = 0;
    long val = std::strtol(str, &endptr, 10);
    if (errno != 0 || *endptr != '\0' || val < INT_MIN || val > INT_MAX) {
        return false;
    }
    out = static_cast<int>(val);
    return true;
}
#endif //CUDASAMPLE_FUNCTIONS_H
