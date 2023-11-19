import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_save_corners(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners using Harris Corner Detector
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Dilate the result to mark the corners
    dst = cv2.dilate(dst, None)

    # Threshold to get the corners and enlarge them
    img[dst > 0.01 * dst.max()] = [255,0,0]

    # Save the image with corners
    cv2.imwrite(output_path, img)

    # Displaying the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corners Detected")
    plt.show()

# Replace 'path_to_image.jpg' and 'output_path.jpg' with your image file and desired output file
image_path = '/Users/treblocami/Desktop/job/cs484_project/hw4_2023f/questions/Chase2.jpg'
output_path = '/Users/treblocami/Desktop/job/cs484_project/hw4_2023f/questions/RISHLibrary1_corners.jpg'
detect_and_save_corners(image_path, output_path)
