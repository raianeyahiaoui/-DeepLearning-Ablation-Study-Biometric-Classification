# --- Start of B_Haar_Cascade_Detection.py Content ---

# 1. Setup and Imports
import cv2
import matplotlib.pyplot as plt
import urllib.request
import os

# Define the standard path for the OpenCV Haar Cascade XML file
# We use a public URL to download the file directly, making the script self-contained.
HAARCASCADE_EYE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
HAARCASCADE_EYE_FILE = "haarcascade_eye.xml"

# Define image paths (These assume you created the 'data/' folder and uploaded your images)
IMAGE_PATH_UBIRIS = "../data/S5001L04.jpg" 
IMAGE_PATH_CASIA = "../data/C28_S1_I11.tiff" 
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Download the cascade file if it doesn't exist (ensures the script runs anywhere)
if not os.path.exists(HAARCASCADE_EYE_FILE):
    print(f"Downloading {HAARCASCADE_EYE_FILE}...")
    try:
        urllib.request.urlretrieve(HAARCASCADE_EYE_URL, HAARCASCADE_EYE_FILE)
    except Exception as e:
        print(f"Error downloading cascade: {e}. Please ensure you have network access.")

# Load the cascade classifier
eye_cascade = cv2.CascadeClassifier(HAARCASCADE_EYE_FILE)
if eye_cascade.empty():
    print("Error: Haar Cascade file failed to load.")
    # Exit or handle the error gracefully if it can't load

# 2. Define the Detection and Visualization Function
def detect_and_visualize_eyes(image_path, cascade_classifier):
    """
    Loads an image, detects eyes using Haar Cascade (using tuned parameters), 
    and visualizes the detected regions as a proof of concept for ROI localization.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}. Skipping visualization.")
        return

    original_for_display = image.copy()
    
    # Haar Cascade requires a grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes using tuned parameters (from your research/Item 13.2)
    # minNeighbors=7 is a demonstration of parameter tuning for noise reduction.
    eyes = cascade_classifier.detectMultiScale(
        gray_image, 
        scaleFactor=1.1, 
        minNeighbors=7, # Tuned parameter for reducing false positives
        minSize=(30, 30)
    )

    # Draw rectangles on the original image (Visualization like your plots)
    processed_image = original_for_display.copy()
    for (x, y, w, h) in eyes:
        cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 0, 255), 2) # Bounding box is Blue

    # Setup the plot figure
    num_eyes = len(eyes)
    fig, axs = plt.subplots(1, num_eyes + 1, figsize=(4 * (num_eyes + 1), 5))
    fig.suptitle(f'Haar Cascade Eye Detection & ROI Zoom for: {image_path.split("/")[-1]}', fontsize=16)

    # --- Plot 1: The processed image with detection boxes ---
    axs[0].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f'Detected Eyes ({num_eyes} regions)')
    axs[0].axis('off')

    # --- Plot 2+: The zoomed-in and resized detected eyes ---
    for i, (x, y, w, h) in enumerate(eyes):
        zoomed_eye = original_for_display[y:y+h, x:x+w]
        zoomed_eye_resized = cv2.resize(zoomed_eye, (IMG_WIDTH, IMG_HEIGHT))
        
        axs[i+1].imshow(cv2.cvtColor(zoomed_eye_resized, cv2.COLOR_BGR2RGB))
        axs[i+1].set_title(f'Eye {i+1} (DCNN Input Size)')
        axs[i+1].axis('off')
    
    # Hide unused axes
    for j in range(num_eyes + 1, 4):
         if j < len(axs):
             axs[j].axis('off')
    
    plt.show()
    print(f"Successfully detected {num_eyes} eye regions.")


# 3. Execution
if not eye_cascade.empty():
    print("--- Running on UBIRISV2 Example ---")
    detect_and_visualize_eyes(IMAGE_PATH_UBIRIS, eye_cascade)

    print("\n--- Running on CASIA Example ---")
    detect_and_visualize_eyes(IMAGE_PATH_CASIA, eye_cascade)

# --- End of B_Haar_Cascade_Detection.py Content ---
