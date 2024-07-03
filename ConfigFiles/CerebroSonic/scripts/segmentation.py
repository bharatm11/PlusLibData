# This script reads an image and processes it to segment out noise using simple thresholding.
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

# Read the image
image = cv2.imread('falx.jpg')
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth the image using Gaussian filter
smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 2)

# Display the smoothed image
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image')
plt.show()

# Binarize the image using Otsu's thresholding
_, binary_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.show()

# Edge detection using Sobel, Canny, and Prewitt methods
sobel_edges = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 1, ksize=3)
sobel_edges = np.uint8(np.absolute(sobel_edges))
canny_edges = cv2.Canny(smoothed_image, 100, 200)

# Prewitt operator (custom implementation)
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewitt_edges_x = cv2.filter2D(smoothed_image, -1, kernelx)
prewitt_edges_y = cv2.filter2D(smoothed_image, -1, kernely)
prewitt_edges = prewitt_edges_x + prewitt_edges_y

# Combine edges
combined_edges = np.bitwise_or(sobel_edges, canny_edges)
combined_edges = np.bitwise_or(combined_edges, prewitt_edges)

# Combine the binary image with the combined edges
combined_image = np.bitwise_or(binary_image, combined_edges)

# Display the combined image
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Image')
plt.show()

# Masked image
masked_image = gray_image.copy()
masked_image[binary_image == 0] = 0

# Display original and masked images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(masked_image, cmap='gray')
ax[1].set_title('Masked Image')
ax[1].axis('off')
plt.show()
