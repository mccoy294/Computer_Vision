import cv2
import numpy as np
import torch

# Load the image
image_path = '/Users/ryanmccoy/OCR_app/Computer_Vision/Building_Supplies/data/sheetrock/certainteed-drywall.jpg'
image = cv2.imread(image_path)

# Get the dimensions of the image
image_height, image_width, _ = image.shape

# Define the crop dimensions
crop_left = 0
crop_right = image_width
crop_top = 0
crop_bottom = image_height

# Crop the specified region
cropped_region = image[crop_top:crop_bottom, crop_left:crop_right]

# Convert the cropped region to grayscale
gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection using Sobel Y filter
edges_canny = cv2.Canny(gray, threshold1=50, threshold2=150)

# Find the indices where there are foreground edges
edge_indices = np.where(edges_canny > 0)

# Map Sobel Y edges to the original image and extract corresponding scalar values
scalar_values = gray[edge_indices]

# Reshape scalar values into a 1D array
scalar_values_1d = scalar_values.reshape(-1)

# Overlay Canny edges on the original image
overlay = image.copy()
overlay[crop_top:crop_bottom, crop_left:crop_right][edges_canny > 0] = [0, 255, 0]  # Set edge pixels to green (adjust color as needed)

# Display the overlayed image
"""cv2.imshow("Overlayed Canny Edges", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Print the extracted scalar values
print("Scalar values from the original image:\n", scalar_values_1d)

# Count the number of edges
edge_count = np.count_nonzero(edges_canny)
print("Number of edges detected:", edge_count)