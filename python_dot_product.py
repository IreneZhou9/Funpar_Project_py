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
@timer
def py_1d_dot(arr1, arr2):
    return sum(x * y for x, y in zip(arr1, arr2))

@timer
def np_1d_dot(arr1, arr2):
    return np.dot(arr1, arr2)

@timer
def rust_1d_dot(arr1, arr2):
    return dotpro.dot_product(arr1, arr2)

@timer
def rust_1d_dot_any(arr1, arr2):
    return dotpro.dot_product_any(arr1, arr2)

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
    return np.array(arr1).dot(arr2)

@timer
def rust_2d_dot(arr1, arr2):
    return dotpro.matrix_multiply(arr1, arr2)

if __name__ == '__main__':

    # Benchmarking dot product of 1D array
    # arr1 = generate_1d_array(700000, (1, 100))
    # arr2 = generate_1d_array(700000, (1, 100))
    # arr1 = np.random.rand(50000000)
    # arr2 = np.random.rand(50000000)
    # arr1_list = arr1.tolist()
    # arr2_list = arr2.tolist()
    #
    # numpy_time_taken, numpy_result = np_1d_dot(arr1, arr2)
    # python_time_taken, python_result = py_1d_dot(arr1, arr2)
    # rust_time_taken, rust_result = rust_1d_dot(arr1_list, arr2_list)
    # rust_any_time_taken, rust_any_result = rust_1d_dot_any(arr1, arr2)
    #
    # print(f"dot product time taken with rust: {rust_time_taken:.6f} seconds, result is {rust_result}")
    # print(f"dot product time taken with rust any: {rust_any_time_taken:.6f} seconds, result is {rust_any_result}")
    # print(f"dot product time taken with python: {python_time_taken:.6f} seconds, result is {python_result}")
    # print(f"dot product time taken with numpy: {numpy_time_taken:.6f} seconds, result is {numpy_result}")
    # Generate large random float arrays
    # ..................................................................
    # arr1 = np.random.randint(10000, size=50000000)
    # arr2 = np.random.randint(10000, size=50000000)
    #
    # # Convert NumPy arrays to lists
    # start_time = time.time()
    # arr1_list = arr1.tolist()
    # arr2_list = arr2.tolist()
    # end_time = time.time()
    # conversion_time = end_time - start_time
    # print(f"Time taken for conversion to list: {conversion_time:.6f} seconds")
    #
    # # Benchmark the function that works with Vec<f64>
    # rust_time_taken, rust_result = rust_1d_dot(arr1_list, arr2_list)
    # print(f"dot product time taken with rust (Vec<i64>): {rust_time_taken:.6f} seconds, result is {rust_result}")
    #
    # # Benchmark the function that works with PyAny (directly with NumPy arrays)
    # rust_any_time_taken, rust_any_result = rust_1d_dot_any(arr1, arr2)
    # print(f"dot product time taken with rust any: {rust_any_time_taken:.6f} seconds, result is {rust_any_result}")
    #
    # # Benchmark the function in Python
    # python_time_taken, python_result = py_1d_dot(arr1, arr2)
    # print(f"dot product time taken with python: {python_time_taken:.6f} seconds, result is {python_result}")
    #
    # # Benchmark the function in NumPy
    # numpy_time_taken, numpy_result = np_1d_dot(arr1, arr2)
    # print(f"dot product time taken with numpy: {numpy_time_taken:.6f} seconds, result is {numpy_result}")


    # Benchmarking matrix multiplication
    data = generate_2d_array(size=500, range_=(1, 100))
    numpy_time_taken, numpy_result = np_2d_dot(*data)
    python_time_taken, python_result = py_2d_dot(*data)
    print(f"time taken with numpy: {numpy_time_taken} seconds")
    print(f"time taken with python: {python_time_taken} seconds")

