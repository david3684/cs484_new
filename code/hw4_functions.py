import cv2
import numpy as np


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
    img_gray = np.float32(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    Ix = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), 0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), 0)
    
    # Compute the Harris response
    detM = Ixx * Iyy - Ixy * Ixy
    traceM = Ixx + Iyy
    alpha = 0.04
    cornerness = detM - alpha * traceM * traceM
    threshold = 0.01 * cornerness.max()
    non_max_sup_img = np.zeros_like(cornerness)
    non_max_sup_img[cornerness > threshold] = cornerness[cornerness > threshold]
    
    # Apply non-maximum suppression
    corners = cv2.dilate(non_max_sup_img, None)
    x, y = np.where(corners == non_max_sup_img)
    edge_threshold = descriptor_window_image_width // 2
    x, y = x[(x >= edge_threshold) & (x < image.shape[1] - edge_threshold)], y[(y >= edge_threshold) & (y < image.shape[0] - edge_threshold)]

    return x, y

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
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pre-compute gradients and orientations
    # These could also be passed in as arguments if computed earlier
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360

    # Set up the SIFT-like feature descriptors array
    descriptor_length = 128  # This is the standard SIFT descriptor length
    features = np.zeros((len(x), descriptor_length))

    # Define the number of bins for the histogram
    bin_width = 360 // (descriptor_length // 16)  # Assuming a 4x4 grid

    # Descriptor extraction
    cell_width = descriptor_window_image_width // 4
    border_width = descriptor_window_image_width // 2
    
    for i in range(len(x)):
        # For each interest point, build a histogram of gradients
        hist = np.zeros(descriptor_length)
        for row in range(-border_width, border_width, cell_width):
            for col in range(-border_width, border_width, cell_width):
                sub_magnitude = magnitude[y[i]+row:y[i]+row+cell_width, x[i]+col:x[i]+col+cell_width]
                sub_orientation = orientation[y[i]+row:y[i]+row+cell_width, x[i]+col:x[i]+col+cell_width]
                
                # Weighted contribution to histogram bins
                for m, o in zip(sub_magnitude.flatten(), sub_orientation.flatten()):
                    bin = int(o // bin_width)
                    hist[bin] += m  # Could add weighting by distance from center of descriptor
        
        # Normalize the feature descriptor to ensure contrast invariance
        hist /= (np.linalg.norm(hist) + 1e-10)  # Adding a small value to avoid division by zero
        
        # Clip the values to [0, 0.2] to reduce effects of lighting changes
        hist = np.clip(hist, 0, 0.2)
        
        # Renormalize again
        hist /= (np.linalg.norm(hist) + 1e-10)

        # Assign the histogram to the feature descriptor array
        features[i, :] = hist.flatten()

    return features

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
    num_features = min(features1.shape[0], features2.shape[0])
    matches = np.zeros((num_features, 2))
    matches[:,0] = np.random.permutation(num_features)
    matches[:,1] = np.random.permutation(num_features)
    confidences = np.random.rand(num_features)
    return matches, confidences

