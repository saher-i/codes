# Re-importing matplotlib.pyplot after code execution state reset
import matplotlib.pyplot as plt

# Given data for the number of workers and the corresponding time taken
worker_counts = [0, 4, 8, 12, 16, 20, 24, 28, 32]
times = [18.21, 5.39, 4.24, 4.32, 4.51, 4.82, 4.59, 4.71, 4.84 ]

# Plotting function
def plot_workers_vs_time(worker_counts, times):
    plt.figure(figsize=(10, 6))
    plt.plot(worker_counts, times, marker='o', linestyle='-', color='b')
    plt.title('DataLoader Performance vs. Number of Workers')
    plt.xlabel('Number of Workers')
    plt.ylabel('Total Time Taken (seconds)')
    plt.xticks(worker_counts)
    plt.grid(True)
    plt.show()

# Call the plotting function with the provided data
plot_workers_vs_time(worker_counts, times)

