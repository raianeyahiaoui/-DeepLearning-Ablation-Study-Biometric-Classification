# C_SIFT_LBP_Feature_Visualization.ipynb

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern # For LBP (Item 9)

# Define paths for demonstration images
IMAGE_PATH_UBIRIS = "../data/S5001L04.jpg" 
IMAGE_PATH_CASIA = "../data/C28_S1_I11.tiff" 
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Function to load and preprocess the image
def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return resized_image, gray_image

# --- SIFT Keypoint Visualization ---

def visualize_sift(image_path):
    color_img, gray_img = load_and_preprocess(image_path)
    
    # Initialize SIFT detector (Using your best tuned parameters from the thesis)
    # The parameters (nfeatures=1000, contrastThreshold=0.01, edgeThreshold=5) 
    # demonstrate deliberate hyperparameter tuning for iris texture.
    sift = cv2.SIFT_create(
        nfeatures=1000, 
        contrastThreshold=0.01, 
        edgeThreshold=5, 
        sigma=1.2 # Your chosen sigma
    )

    # Detect keypoints
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    
    # Draw keypoints (Item 17 visualization)
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS shows scale and orientation.
    image_with_keypoints = cv2.drawKeypoints(
        color_img, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Keypoint Detection: {image_path.split("/")[-1]} ({len(keypoints)} Keypoints)')
    plt.axis('off')
    plt.show()
    
# Run SIFT visualization on both datasets
visualize_sift(IMAGE_PATH_UBIRIS)
visualize_sift(IMAGE_PATH_CASIA)

# --- Local Binary Patterns (LBP) Feature Visualization ---

def visualize_lbp(image_path):
    color_img, gray_img = load_and_preprocess(image_path)

    # LBP parameters (P: points, R: radius)
    P = 24  # Number of neighbors
    R = 8   # Radius
    
    # Compute LBP features (Item 9 logic)
    lbp_features = local_binary_pattern(gray_img, P, R, method='uniform')

    plt.figure(figsize=(10, 5))
    
    # 1. Plot the SIFT-annotated image (Context)
    # Re-running SIFT here just to draw the rich keypoints for context
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.01, edgeThreshold=5, sigma=1.2)
    keypoints, _ = sift.detectAndCompute(gray_img, None)
    image_with_keypoints = cv2.drawKeypoints(color_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f'Image with SIFT Keypoints')
    plt.axis('off')
    
    # 2. Plot the LBP Texture Map
    plt.subplot(1, 2, 2)
    plt.imshow(lbp_features, cmap='gray')
    plt.title(f'Local Binary Patterns (LBP) Texture Map')
    plt.axis('off')
    
    plt.suptitle(f'Traditional Feature Comparison for: {image_path.split("/")[-1]}', fontsize=16)
    plt.show()

# Run LBP visualization
visualize_lbp(IMAGE_PATH_UBIRIS)
