import matplotlib.pyplot as plt
import numpy as np

# Constants
peak_performance = 200  # GFLOPS
memory_bandwidth = 30  # GB/s
arithmetic_intensity = (
    peak_performance / memory_bandwidth
)  # Example value, this can be adjusted based on specific requirements

# Create a range of arithmetic intensities
ai_range = np.linspace(0.1, 10000, 100)

# Calculate performance for each arithmetic intensity
performance = np.minimum(peak_performance, memory_bandwidth * ai_range)

# New values to add to the plot
values_to_add = [
    (4.126097e02, 1.031524e11, "c1l"),
    (8.261685e02, 2.065421e11, "c2l"),
    (3.583381e03, 8.958451e11, "c3l"),
    (0.020399613737716114, 2549951.7172145145, "c4l"),
    (5.203956201975509, 650494525.2469386, "c5l"),
    (2.093976e04, 5.234941e12, "c1s"),
    (6.721675e04, 1.680419e13, "c2s"),
    (3.541190e05, 8.852975e13, "c3s"),
    (0.019884857933863832, 2485607.241732979, "c4s"),
    (5.238671462020795, 654833932.7525994, "c5s"),
]

# Convert FLOPS to GFLOPS for consistency
values_to_add = [(b, f / 1e9, l) for (b, f, l) in values_to_add]

# Calculate arithmetic intensity for each point
ai_values = [f / b for (b, f, l) in values_to_add]

# Plotting the roofline model again
plt.figure(figsize=(10, 6))
plt.plot(ai_range, performance, label="Roofline", color="blue")
plt.axvline(
    x=arithmetic_intensity, color="red", linestyle="--", label="Arithmetic Intensity"
)

# Adding the new points to the plot
for (b, f, l), ai in zip(values_to_add, ai_values):
    plt.plot(ai, f, "o", label=f"{l}")

# Adding a horizontal line for peak performance
plt.axhline(y=peak_performance, color="gray", linestyle="--")
plt.text(
    0.1,
    peak_performance,
    f"Peak Performance = {peak_performance} GFLOPS",
    verticalalignment="bottom",
)

# Annotations and labels
plt.title("Roofline Model with Points")
plt.xlabel("Arithmetic Intensity (FLOPS/Byte)")
plt.ylabel("Performance (GFLOPS)")
plt.xscale("log")
plt.yscale("log")
# plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
