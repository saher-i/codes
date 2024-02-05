import numpy as np
import time
import argparse
import tqdm


def dp(A, B):
    R = 0.0
    for j in range(A.size):
        R += A[j] * B[j]
    return R


def benchmark(N, repetitions):
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    exec_time = []

    for _ in tqdm.tqdm(range(0, repetitions//2)):
        dp(A, B)
    for _ in tqdm.tqdm(range(repetitions//2, repetitions)):
        start_time = time.monotonic_ns()
        dp(A, B)
        end_time = time.monotonic_ns()
        exec_time.append(end_time - start_time)

    mean_time_ns = np.mean(exec_time)
    mean_time_s = mean_time_ns / 1e9

    bytes_accessed = 2 * N * 4
    bandwidth = (bytes_accessed / mean_time_s) / 1e9
    throughput = N / mean_time_s

    return mean_time_s, bandwidth, throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dot Product Performance Measurement")
    parser.add_argument("N", type=int, help="Size of the vectors")
    parser.add_argument(
        "repetitions", type=int, help="Number of repetitions for the measurement"
    )

    args = parser.parse_args()

    mean_time, bandwidth, throughput = benchmark(args.N, args.repetitions)
    print(
        f"N: {args.N}: <T>: {mean_time}s, B: {bandwidth}GB/s, T:  {throughput}FLOP/s"
    )
