import numpy as np
print("\n--- DETAILED CONVOLUTION EXAMPLE ---")

# Create a small 4x4 grayscale image and a 3x3 kernel
small_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

small_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

print("Small Image:")
print(small_image)
print("\nKernel (Vertical Edge Detection):")
print(small_kernel)

# Perform convolution manually for demonstration
output_height = small_image.shape[0] - small_kernel.shape[0] + 1
output_width = small_image.shape[1] - small_kernel.shape[1] + 1
manual_output = np.zeros((output_height, output_width))

# print(small_image[0:0+small_kernel.shape[0], 0:0+small_kernel.shape[1]])

for i in range(output_height):
    for j in range(output_width):
        # Extract window from the image
        window = small_image[i:i+small_kernel.shape[0], j:j+small_kernel.shape[1]]
        # Element-wise multiply and sum
        manual_output[i, j] = np.sum(window * small_kernel)

print("\nManual Convolution Result:")
print(manual_output)

# Let's show the step-by-step calculation for the first output element (0,0)
print("\nDetailed calculation for output[0,0]:")
window = small_image[0:3, 0:3]
print("Window from image:")
print(window)
print("\nElement-wise multiplication with kernel:")
print(window * small_kernel)
print(f"\nSum of element-wise multiplication: {np.sum(window * small_kernel)}")
