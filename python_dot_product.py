import time
import numpy as np
import random as rn
import dotpro

# Timer Function
def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()
        return after - before, result

    return wrapper

# 1D array dot product
def generate_1d_array(size, range_):
    arr = [rn.randrange(*range_) for _ in range(size)]
    return arr

@timer
def py_1d_dot(arr1, arr2):
    return sum(x * y for x, y in zip(arr1, arr2))
@timer
def np_1d_dot(arr1, arr2):
    return np.dot(arr1, arr2)

@timer
def rust_1d_dot(arr1, arr2):
    return dotpro.dot_product(arr1, arr2)

# 2D matrix dot product

def generate_2d_array(size, range_):
    arr = [[[rn.randrange(*range_) for _ in range(size)] for _ in range(size)] for _ in range(2)]
    return arr

@timer
def py_2d_dot(arr1, arr2):
    result = [[0 for _ in range(len(arr1))] for _ in range(len(arr2[0]))]
    for i in range(len(arr1)):
        for j in range(len(arr2[0])):
            for k in range(len(arr2[0])):
                result[i][j] += arr1[i][k] * arr2[k][j]
    return result

@timer
def np_2d_dot(arr1, arr2):
    return np.matmul(arr1, arr2)

@timer
def rust_2d_dot(arr1, arr2):
    return dotpro.matrix_multiply(arr1, arr2)

if __name__ == '__main__':

    # Benchmarking dot product of 1D array
    arr1 = generate_1d_array(60000, (1, 100))
    arr2 = generate_1d_array(60000, (1, 100))

    numpy_time_taken, numpy_result = np_1d_dot(arr1, arr2)
    python_time_taken, python_result = py_1d_dot(arr1, arr2)
    rust_time_taken, rust_result = rust_1d_dot(arr1, arr2)

    print(f"dot product time taken with python: {python_time_taken:.6f} seconds, result is {python_result}")
    print(f"dot product time taken with numpy: {numpy_time_taken:.6f} seconds, result is {numpy_result}")
    print(f"dot product time taken with rust: {rust_time_taken:.6f} seconds, result is {rust_result}")

    # Benchmarking matrix multiplication
    data = generate_2d_array(size=500, range_=(1, 100))

    numpy_time_taken, numpy_result = np_2d_dot(*data)
    python_time_taken, python_result = py_2d_dot(*data)
    rust_time_taken, rust_result = rust_2d_dot(*data)

    print(f"matrix multiplication time taken with numpy: {numpy_time_taken} seconds")
    print(f"matrix multiplication time taken with python: {python_time_taken} seconds")
    print(f"matrix multiplication time taken with rust: {rust_time_taken} seconds")

