# --- Start of A_Initial_Enhancement_Pipeline.ipynb Content ---

# 1. Setup and Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define standard DCNN input size
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define paths for demonstration images
# NOTE: These image files must be uploaded to your GitHub repository (e.g., in a 'data/' folder)
IMAGE_PATH_UBIRIS = "../data/S5001L04.jpg" 
IMAGE_PATH_CASIA = "../data/C28_S1_I11.tiff" 

# 2. Define the Preprocessing Pipeline Function
def process_and_normalize_image(image_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """
    Applies the image enhancement pipeline: Resize, Gaussian Denoising, and Normalization.
    This prepares heterogeneous (visible-light/NIR) data for DCNN input.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return None, None, None, None

    # Step 1: Resize to DCNN standard input
    resized_image = cv2.resize(image, (img_width, img_height))

    # Step 2: Apply Gaussian Blur for Denoising (Kernel 5x5)
    # This reduces sensor noise and mild motion blur present in 'in-the-wild' images.
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Step 3: Normalize to [-1, 1] range (Standard for ResNet/MobileNet preprocessing)
    # The original formula from your code was (x / 127.5) - 1.
    normalized_image = (blurred_image / 127.5) - 1

    return image, resized_image, blurred_image, normalized_image

# 3. Define the Visualization Function
def visualize_pipeline(img_path):
    original, resized, blurred, normalized = process_and_normalize_image(img_path)

    if original is None:
        return

    # To visualize the normalized image, we must scale it back to 0-255 range.
    normalized_vis = ((normalized + 1) * 127.5).astype(np.uint8)

    # Plot the 4-panel result
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Initial Enhancement Pipeline for: {img_path.split("/")[-1]}', fontsize=16)

    # Note: cv2 reads in BGR format, so convert to RGB for correct matplotlib display
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f'Resized ({IMG_WIDTH}x{IMG_HEIGHT})')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Gaussian Blurred (Denoised)')
    axs[2].axis('off')

    axs[3].imshow(cv2.cvtColor(normalized_vis, cv2.COLOR_BGR2RGB))
    axs[3].set_title('Normalized [-1, 1] for DCNN Input')
    axs[3].axis('off')

    plt.show()
    print("\n--- Summary of Results ---")
    print(f"Original shape: {cv2.imread(img_path).shape}")
    print(f"Final DCNN Input shape (Normalized): {normalized.shape}, Value Range: [{normalized.min()}, {normalized.max()}]")


# 4. Execution
print("--- Running Pipeline on UBIRISV2 Example (Visible Light) ---")
visualize_pipeline(IMAGE_PATH_UBIRIS)

print("\n--- Running Pipeline on CASIA Example (NIR/Grayscale) ---")
visualize_pipeline(IMAGE_PATH_CASIA)

# --- End of A_Initial_Enhancement_Pipeline.ipynb Content ---
