import random
import math
import matplotlib.pyplot as plt
import numpy as np


def _size(k, min_value, max_value):
    if k <= 1:
        return min_value

    range_size = max_value - min_value + 1
    p = 1 / k

    u = random.random()
    x = math.log(1 - u * (1 - (1 - p) ** range_size)) / math.log(1 - p)
    return min_value + math.floor(x)


min_value = 1
average_size = 10
max_value = 40
num_samples = 100000


# Generate samples
samples = [_size(average_size, min_value, max_value) for _ in range(num_samples)]

# Calculate actual average
actual_average = np.mean(samples)

print(f"Desired average: {average_size}")
print(f"Actual average: {actual_average:.2f}")
print(f"Min value: {min_value}")
print(f"Max value: {max_value}")

# Create histogram
plt.figure(figsize=(12, 6))
counts, bins, _ = plt.hist(
    samples, bins=range(min_value, max_value + 2), align="left", rwidth=0.8
)
plt.title(
    f"Bounded Geometric Distribution (k={average_size}, min={min_value}, max={max_value})"
)
plt.xlabel("Value")
plt.ylabel("Frequency")

# Add vertical lines for desired and actual averages
plt.axvline(
    x=average_size, color="r", linestyle="--", label=f"Desired Average ({average_size})"
)
plt.axvline(
    x=actual_average,
    color="g",
    linestyle="--",
    label=f"Actual Average ({actual_average:.2f})",
)

# Add percentage labels on top of each bar
for i, count in enumerate(counts):
    percentage = count / num_samples * 100
    plt.text(i + min_value, count, f"{percentage:.1f}%", ha="center", va="bottom")

plt.legend()
plt.grid(True, alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()

# Distribution of values
print("\nDistribution of values:")
for i in range(min_value, max_value + 1):
    count = samples.count(i)
    print(f"Value {i}: {count} ({count/num_samples*100:.2f}%)")
