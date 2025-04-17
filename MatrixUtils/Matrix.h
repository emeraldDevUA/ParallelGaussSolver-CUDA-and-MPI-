#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>

template <typename T>
class Matrix {
public:
    // Constructors
    Matrix(int rows, int cols, T initValue = T());

    // Element access
    T& operator()(int row, int col);
    const T& operator()(int row, int col) const;

    // Matrix dimensions
    int rows() const;
    int cols() const;

    // Operators
    Matrix<T>& operator+=(const Matrix<T>& matrix);

    // Utility methods
    void display() const;
    std::vector<T> getRow(int row) const;
    std::vector<T> getCol(int col) const;

    // Matrix operations
    static Matrix<T> multiply(const Matrix<T>& m1, const Matrix<T>& m2);
    Matrix<T> subMatrix(int colStart, int rowStart, int colEnd, int rowEnd);
    void addToBlock(const Matrix<T>& block, int start_row, int start_col);
    T sum();
    void fillRand(T max);

    T * toArray() {
        T* array = static_cast<T*>(malloc(sizeof(T) * rows() * cols()));
        if (!array) {
            throw std::bad_alloc();
        }

        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) {
                array[i * cols() + j] = data[i][j];
            }
        }
        return array;
    }

private:
    std::vector<std::vector<T>> data;

    // Helper
    static T column_X_row(const std::vector<T>& column, const std::vector<T>& row);
};



// --- Implementation ---

template <typename T>
Matrix<T>::Matrix(int rows, int cols, T initValue)
        : data(rows, std::vector<T>(cols, initValue)) {}

template <typename T>
T& Matrix<T>::operator()(int row, int col) {
    if (row < 0 || row >= rows() || col < 0 || col >= cols()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[row][col];
}

template <typename T>
const T& Matrix<T>::operator()(int row, int col) const {
    if (row < 0 || row >= rows() || col < 0 || col >= cols()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[row][col];
}

template <typename T>
int Matrix<T>::rows() const {
    return data.size();
}

template <typename T>
int Matrix<T>::cols() const {
    return data.empty() ? 0 : data[0].size();
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& matrix) {
    if (rows() != matrix.rows() || cols() != matrix.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            data[i][j] += matrix(i, j);
        }
    }
    return *this;
}

template <typename T>
void Matrix<T>::display() const {
    for (const auto& row : data) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }
}

template <typename T>
std::vector<T> Matrix<T>::getRow(int row) const {
    if (row < 0 || row >= rows()) {
        throw std::out_of_range("Row index out of range");
    }
    return data[row];
}

template <typename T>
std::vector<T> Matrix<T>::getCol(int col) const {
    if (col < 0 || col >= cols()) {
        throw std::out_of_range("Column index out of range");
    }
    std::vector<T> column;
    column.reserve(rows());
    for (const auto& row : data) {
        column.push_back(row[col]);
    }
    return column;
}

template <typename T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& m1, const Matrix<T>& m2) {
    if (m1.cols() != m2.rows()) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }

    Matrix<T> result(m1.rows(), m2.cols(), T());

    std::vector<std::vector<T>> m2Cols(m2.cols());
    for (int j = 0; j < m2.cols(); ++j) {
        m2Cols[j] = m2.getCol(j);
    }

    for (int i = 0; i < m1.rows(); ++i) {
        std::vector<T> m1Row = m1.getRow(i);
        for (int j = 0; j < m2.cols(); ++j) {
            result(i, j) = column_X_row(m1Row, m2Cols[j]);
        }
    }

    return result;
}

template <typename T>
T Matrix<T>::column_X_row(const std::vector<T>& column, const std::vector<T>& row) {
    T result = T();
    for (size_t i = 0; i < column.size(); ++i) {
        result += column[i] * row[i];
    }
    return result;
}


template<typename T>
Matrix<T> Matrix<T>::subMatrix(int colStart, int rowStart, int colEnd, int rowEnd) {
    if (rowStart < 0 || rowEnd > rows() || colStart < 0 || colEnd > cols() ||
        rowStart >= rowEnd || colStart >= colEnd) {
        throw std::out_of_range("Invalid submatrix range");
    }

    Matrix<T> finalMat(rowEnd - rowStart, colEnd - colStart);
    for (int i = rowStart; i < rowEnd; ++i) {
        for (int j = colStart; j < colEnd; ++j) {
            finalMat(i - rowStart, j - colStart) = data[i][j];
        }
    }

    return finalMat;
}

template<typename T>
void Matrix<T>::addToBlock(const Matrix<T>& block, int start_row, int start_col) {
    if (start_row + block.rows() > this->rows() || start_col + block.cols() > this->cols()) {
        throw std::out_of_range("Block dimensions exceed matrix bounds.");
    }

    for (int i = 0; i < block.rows(); ++i) {
        for (int j = 0; j < block.cols(); ++j) {
            (*this)(start_row + i, start_col + j) += block(i, j);
        }
    }
}

template <typename T>
T Matrix<T>::sum() {
    T result = T();
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            result += data[i][j];
        }
    }
    return result;
}

template <typename T>
void Matrix<T>::fillRand(T max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-max, max);

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            (*this)(i, j) = dist(gen);
        }
    }
}
