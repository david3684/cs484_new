import cv2
import numpy as np
import skimage
from scipy.ndimage import convolve

def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    # Placeholder that you can delete -- random points
    #x = np.floor(np.random.rand(500) * np.float32(image.shape[1]))
    #y = np.floor(np.random.rand(500) * np.float32(image.shape[0]))
    #return x,y

    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000
    # Convert image to grayscale if it is color
    img_gray = np.float32(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # Define horizontal and vertical gradient filters
    gradient_filter_x = np.array([[-1, 0, 1]])
    gradient_filter_y = np.array([[-1], [0], [1]])

    # Compute gradients by convolving gradient filters with the image
    Ix = cv2.filter2D(src=img_gray, kernel=gradient_filter_x, ddepth=-1)
    Iy = cv2.filter2D(src=img_gray, kernel=gradient_filter_y, ddepth=-1)

    # Compute products of derivatives
    sigma = 1
    alpha = 0.05
    threshold = 0.01
    Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), sigma)
    Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), sigma)
    cornerness = (Ixx*Iyy) - (Ixy**2) - alpha*((Ixx+Iyy)**2)
    corners = cornerness > threshold * cornerness.max()
    distance = 1
    for i in range(distance, corners.shape[0] - distance):
        for j in range(distance, corners.shape[1] - distance):
            if corners[i, j]:
                local_max = cornerness[i - distance:i + distance + 1, j - distance:j + distance + 1].max()
                if cornerness[i, j] != local_max:
                    corners[i, j] = False

    # Getting the coordinates of corners
    y, x = np.where(corners)
    return x, y


def compute_dominant_orientation(magnitude, orientation):
    num_bins = 36
    orientation_histogram = np.zeros((num_bins,))
    angle_step = 360.0 / num_bins

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            angle = orientation[i, j] % 360
            bin_idx = int(angle / angle_step)
            orientation_histogram[bin_idx] += magnitude[i, j]

    dominant_bin = np.argmax(orientation_histogram)
    dominant_orientation = dominant_bin * angle_step

    return dominant_orientation

def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)


    # Placeholder that you can delete. Empty features.
    # Convert to grayscale if the image is color
    # Assuming keypoints are provided as a list of cv2.KeyPoint objects
    # descriptor_window_image_width is the width of the SIFT descriptors. By default, it's set to 16 pixels.
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Constants for descriptor
    num_bins = 8
    window_width = descriptor_window_image_width // 4  # width of the 4x4 cells in a window

    # Preallocate the descriptors array
    descriptors = np.zeros((len(x), 128))  # 128 = 16 histograms * 8 bins per histogram

    # Compute gradients using Sobel operator
    gradient_filter_x = np.array([[-1, 0, 1]])
    gradient_filter_y = np.array([[-1], [0], [1]])

    # Compute gradients by convolving gradient filters with the image
    dx = cv2.filter2D(src=image, kernel=gradient_filter_x, ddepth=-1)
    dy = cv2.filter2D(src=image, kernel=gradient_filter_y, ddepth=-1)
    magnitude = np.sqrt(np.add(np.square(dx), np.square(dy)))
    orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360

    # Pre-compute the Gaussian window
    gaussian_window = cv2.getGaussianKernel(descriptor_window_image_width, descriptor_window_image_width / 2)
    gaussian_window = gaussian_window * gaussian_window.T

    # Iterate over all keypoints
    for idx in range(len(x)):
        xi=x[idx]
        yi=y[idx]
        # Ensure the window is fully within the image bounds
        min_x = max(xi - window_width * 2, 0) #이미지 경계와 descriptor window 경계 중 큰걸로
        max_x = min(xi + window_width * 2, image.shape[1])
        min_y = max(yi - window_width * 2, 0)
        max_y = min(yi + window_width * 2, image.shape[0])

        # Calculate the region of the image to be considered for this keypoint
        window_magnitude = magnitude[min_y:max_y, min_x:max_x]
        window_orientation = orientation[min_y:max_y, min_x:max_x]
        # Apply the Gaussian window
        weight_window_magnitude = window_magnitude * gaussian_window[min_y - yi + window_width * 2:max_y - yi + window_width * 2,
                                                                    min_x - xi + window_width * 2:max_x - xi + window_width * 2]

        # Descriptor array, to be filled with values
        descriptor_vector = np.zeros((4, 4, num_bins))

        # Populate the descriptor vector with values
        for i in range(4):
            for j in range(4):
                # Subregion of window, considering the boundaries
                # Subregion of weighted magnitudes and orientations
                subregion_w_mag = weight_window_magnitude[window_width*i:window_width*(i+1), window_width*j:window_width*(j+1)].flatten()
                subregion_orientation = window_orientation[window_width*i:window_width*(i+1), window_width*j:window_width*(j+1)].flatten()

                # Create histogram for this subregion
                hist, _ = np.histogram(subregion_orientation, bins=num_bins, range=(0, 360), weights=subregion_w_mag)
                descriptor_vector[i, j, :] = hist

        # Normalize the descriptor to be scale invariant
        descriptor_vector = descriptor_vector.flatten()
        descriptor_vector /= (np.linalg.norm(descriptor_vector) + 1e-7)

        # Clip values to 0.2 and re-normalize
        descriptor_vector = np.clip(descriptor_vector, 0, 0.2)
        descriptor_vector /= (np.linalg.norm(descriptor_vector) + 1e-7)

        # Assign the flattened descriptor to the descriptors array
        descriptors[idx, :] = descriptor_vector

    return descriptors

def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.

    matches = []
    confidences = []
    ratio = 0.8

    for i in range(features1.shape[0]):

        distances = np.sqrt(np.square(np.subtract(
            features1[i, :], features2)).sum(axis=1))


        sorted_indices = np.argsort(distances)
        closest_neighbor_index = sorted_indices[0]
        second_closest_neighbor_index = sorted_indices[1]


        nn_distance_ratio = distances[closest_neighbor_index] / distances[second_closest_neighbor_index]


        if nn_distance_ratio < ratio:
            #print(i, closest_neighbor_index)
            matches.append([i, closest_neighbor_index])
            confidences.append(1.0 - nn_distance_ratio)

    matches = np.array(matches)
    confidences = np.array(confidences)

    sorted_indices = np.argsort(confidences)
    matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]

    return matches, confidences
