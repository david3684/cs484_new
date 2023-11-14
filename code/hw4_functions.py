import cv2
import numpy as np
import skimage

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
    sigma=1
    img_gray = np.float32(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    #blurred_image = cv2.GaussianBlur(img_gray, (0,0), sigma)
    blurred_image = img_gray
    Ix = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    Ixx = cv2.GaussianBlur(Ix * Ix, (0,0), sigma)
    Iyy = cv2.GaussianBlur(Iy * Iy, (0,0), sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0,0), sigma)
    
    # Compute the Harris response
    traceM = Ixx + Iyy
    alpha = 0.05
    cornerness = Ixx*Iyy-Ixy*Ixy-alpha*traceM

    # Thresholding to get initial corners
    threshold = 0.08  # You may need to adjust this threshold
    corners_thresholded = cornerness > threshold*cornerness.max()
    skimage.measure.label()
    # Step 6: Non-maxima suppression to pick peaks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(corners_thresholded.astype(np.float32), kernel)
    max_corners = (dilated == corners_thresholded) & (corners_thresholded > 0)

    # Apply edge threshold to suppress the gradients/corners near the edges of the image
    border_size = descriptor_window_image_width // 2
    height, width = image.shape
    suppressed_corners = np.zeros_like(cornerness, dtype=np.float32)
    suppressed_corners[border_size:height - border_size, border_size:width - border_size] = max_corners[border_size:height - border_size, border_size:width - border_size]

    # Find final corner coordinates
    final_corners_y, final_corners_x = np.nonzero(suppressed_corners)

    # Print the number of corners
    print(f"Number of corners detected: {len(final_corners_x)}")

    return final_corners_x, final_corners_y



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

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Constants for descriptor
    num_bins = 8
    window_width = descriptor_window_image_width // 4  # width of the 4x4 cells in a window

    # Preallocate the descriptors array
    descriptors = np.zeros((len(x), 128))  # 128 = 16 histograms * 8 bins per histogram

    # Compute gradients using Sobel operator
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(dx, dy)
    orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360

    # Pre-compute the Gaussian window
    gaussian_window = cv2.getGaussianKernel(descriptor_window_image_width, descriptor_window_image_width / 2)
    gaussian_window = gaussian_window * gaussian_window.T

    # Iterate over all keypoints
    for idx in range(len(x)):
        xi, yi = int(x[idx]), int(y[idx])

        # Ensure the window is fully within the image bounds
        min_x = max(xi - window_width * 2, 0)
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
                i_min = max(0, window_width * i - (yi - window_width * 2))
                j_min = max(0, window_width * j - (xi - window_width * 2))
                i_max = min(window_width * (i + 1), window_magnitude.shape[0])
                j_max = min(window_width * (j + 1), window_magnitude.shape[1])

                # Subregion of weighted magnitudes and orientations
                subregion_w_mag = weight_window_magnitude[i_min:i_max, j_min:j_max].flatten()
                subregion_orientation = window_orientation[i_min:i_max, j_min:j_max].flatten()

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

    # Placeholder random matches and confidences.
    matches = []
    confidences = []
    ratio = 0.8
    # Iterate over all features in features1
    for i, feature in enumerate(features1):
        # Calculate the Euclidean distance from this feature to all features in features2
        distances = np.linalg.norm(features2 - feature, axis=1)

        # Sort the distances to get the closest and second closest match
        sorted_distance_indices = np.argsort(distances)
        closest_neighbor_index = sorted_distance_indices[0]
        second_closest_neighbor_index = sorted_distance_indices[1]

        # Compute the nearest neighbor distance ratio
        nn_distance_ratio = distances[closest_neighbor_index] / distances[second_closest_neighbor_index]

        # If the ratio is below the threshold, it's a good match
        if nn_distance_ratio < ratio:
            #print(nn_distance_ratio)
            matches.append([i, closest_neighbor_index])
            confidences.append((1.0 - nn_distance_ratio) * distances[second_closest_neighbor_index])

    # Convert matches and confidences to numpy arrays
    matches = np.array(matches)
    confidences = np.array(confidences)
    # Sort the matches by confidence in descending order
    sorted_indices = np.argsort(-confidences)
    matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]

    return matches, confidences
