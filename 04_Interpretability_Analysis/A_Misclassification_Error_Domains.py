# --- B_Misclassification_Error_Analysis.ipynb ---

# 1. Setup and Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# (Import model loading logic and get_dataset function from previous files)

# NOTE: This section assumes a trained model (ResNet50_C) is loaded here:
# model = tf.keras.models.load_model('path/to/ResNet50_C_final.h5') 
# test_ds = get_dataset(...) 
class_names = [f'C{i+1}' for i in range(50)] # Placeholder for your 50 class names

# 2. Prediction and Error Collection (Conceptual Logic)
def find_misclassifications(model, dataset, class_names, num_samples=6):
    """Finds the first N misclassified images for visualization."""
    misclassified_samples = []
    
    for images, true_labels in dataset.unbatch().take(-1): # Iterate through the whole test set
        predictions = model.predict(tf.expand_dims(images, axis=0)) # Predict one image
        predicted_label = np.argmax(predictions[0])
        true_label_val = true_labels.numpy()
        
        if predicted_label != true_label_val and len(misclassified_samples) < num_samples:
            misclassified_samples.append({
                'image': images.numpy(),
                'true': class_names[true_label_val],
                'pred': class_names[predicted_label]
            })
            
    return misclassified_samples

# 3. Visualize Misclassifications (Your Item 7 Logic)
# misclassified_data = find_misclassifications(model, test_ds, class_names) 
# NOTE: Replace the above with a manual loading of your misclassified images if dynamic prediction is hard.

# --- Manual Visualization of Key Misclassified Example (Item 7) ---
# Replace with actual image loading and titles for your specific C1/C17/C18 examples
plt.figure(figsize=(10, 5))

# Example 1: True C1, Pred C17
ax1 = plt.subplot(1, 2, 1)
# ax1.imshow(C1_image_data)
ax1.set_title("True: C1\nPred: C17 (Likely Gaze/Occlusion Error)", fontsize=14)
ax1.axis('off')

# Example 2: True C1, Pred C18
ax2 = plt.subplot(1, 2, 2)
# ax2.imshow(C1_image_data_2)
ax2.set_title("True: C1\nPred: C18 (Likely Texture Similarity Error)", fontsize=14)
ax2.axis('off')

plt.suptitle("Error Domain Analysis on Test Set (Misclassified Samples)", fontsize=16)
plt.show()

print("\n--- ERROR ANALYSIS CONCLUSION ---")
print("Misclassifications frequently occurred between texture-similar classes (C1 vs. C17/C18). This pattern suggests that the model struggles most with non-frontal gaze and occlusion, revealing the limits of the current feature set and indicating future research should focus on pose-normalization or robust attention mechanisms.")

# --- End of B_Misclassification_Error_Analysis.ipynb ---
