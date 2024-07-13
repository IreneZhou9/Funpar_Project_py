import time
import numpy as np
import random as rn
import dotpro
import c_extension

# Timer Function
def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()
        return after - before, result

    return wrapper

# dot product
@timer
def py_dot(arr1, arr2):
    return sum(x * y for x, y in zip(arr1, arr2))

@timer
def np_dot(arr1, arr2):
    return np.dot(arr1, arr2)

@timer
def rust_dot(arr1, arr2):
    return dotpro.dot_product_par_simd(arr1, arr2)

# matrix multiplication
def generate_matrix(size, range_):
    arr = [[[rn.randrange(*range_) for _ in range(size)] for _ in range(size)] for _ in range(2)]
    return arr

@timer
def py_matmul(arr1, arr2):
    result = [[0 for _ in range(len(arr1))] for _ in range(len(arr2[0]))]
    for i in range(len(arr1)):
        for j in range(len(arr2[0])):
            for k in range(len(arr2[0])):
                result[i][j] += arr1[i][k] * arr2[k][j]
    return result

@timer
def np_matmul(arr1, arr2):
    return np.array(arr1).dot(arr2)

@timer
def c_matmul(arr1, arr2):
    return c_extension.dot_product_optimized_parallel(arr1, arr2)

# Argsort
@timer
def rust_argsort(arr):
    return dotpro.argsort(arr)

@timer
def np_argsort(arr):
    return np.argsort(arr)


if __name__ == '__main__':

    # Benchmarking dot product
    arr1 = np.random.rand(50000000)
    arr2 = np.random.rand(50000000)

    numpy_time_taken, numpy_result = np_dot(arr1, arr2)
    print(f"dot product time taken with numpy: {numpy_time_taken:.6f} seconds, result is {numpy_result}")

    simd_time_taken, simd_result = rust_dot(arr1, arr2)
    print(f"dot product time taken with rust: {simd_time_taken:.6f} seconds, result is {simd_result}")

    # def benchmark_implementations():
    #
    #     total_np_time = 0.0
    #     total_rust_time = 0.0
    #
    #     for _ in range(50):
    #         arr_x = np.random.rand(50000000)
    #         arr_y = np.random.rand(50000000)
    #
    #         np_time, numpy_result = np_dot(arr_x, arr_y)
    #         total_np_time += np_time
    #
    #         rust_time, simd_result = rust_dot(arr_x, arr_y)
    #         total_rust_time += rust_time
    #
    #     print(f"Total dot product time taken with NumPy over 50 runs: {total_np_time:.6f} seconds, average time is {total_np_time/50}")
    #     print(f"Total dot product time taken with Rust parallel SIMD over 50 runs: {total_rust_time:.6f} seconds, average time is {total_rust_time/50}")
    #
    # benchmark_implementations()



    # Benchmarking matrix multiplication
    data = generate_matrix(size=800, range_=(1, 100))
    numpy_time_taken, numpy_result = np_matmul(*data)
    # python_time_taken, python_result = py_matmul(*data)
    c_par_time_taken, c_result = c_matmul(*data)
    # print(f"matrix multiply time taken with python: {python_time_taken} seconds")
    print(f"matrix multiply time taken with numpy: {numpy_time_taken} seconds")
    print(f"matrix multiply time taken with c par: {c_par_time_taken} seconds")

    # Benchmarking argsort
    data = np.random.randint(0, 100000, size=5000000).tolist()

    np_time_taken, mp_result = np_argsort(data)
    print(f"argsort time taken with numpy: {np_time_taken:.6f} seconds")
    rust_time_taken, rust_result = rust_argsort(data)
    print(f"argsort time taken with rust: {rust_time_taken:.6f} seconds")


