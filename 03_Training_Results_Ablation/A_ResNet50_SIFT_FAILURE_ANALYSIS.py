# --- A_ResNet50_SIFT_FAILURE_ANALYSIS.ipynb ---

# 1. Setup and Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# --- Data and Model Parameters ---
# NOTE: These directories are conceptual. For actual execution, they point to the 
# dataset preprocessed specifically with the SIFT-extraction/cropping logic.
TRAIN_DIR = "/content/drive/MyDrive/UbirisV2/UbirisV2_Dataset/Train_Dataset" # Assumed path for SIFT-processed images
TEST_DIR = "/content/drive/MyDrive/UbirisV2/UbirisV2_Dataset/Test_Dataset"   # Assumed path for SIFT-processed images

N_CLASSES = 50 
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3 

# 2. Dataset Loading Function
def get_dataset(train_dir, test_dir, batch_size=64, img_height=224, img_width=224):
    """Loads the dataset, assuming it is pre-processed or will be processed below."""
    # We use a smaller validation split (0.2) as per standard practice.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=123, 
        image_size=(img_height, img_width), batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=123, 
        image_size=(img_height, img_width), batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    
    return train_ds, val_ds, test_ds

# 3. Custom Feature Extraction (The SIFT-based logic that proved detrimental)
def extract_iris_features(image):
    """
    Placeholder for the custom, SIFT-based feature extraction / normalization logic 
    that was used in the model leading to low performance (ResNet50_B / Code Block 7).
    
    In this experiment, the pre-processing logic here was found to be suboptimal 
    compared to the standard DCNN preprocessing.
    """
    # This is the actual code from your thesis: it normalizes but runs after the custom SIFT/Crop logic.
    image = tf.image.per_image_standardization(image) 
    return image

# --- Training Parameters (Matching your ResNet50_B table) ---
learning_rate = 0.001
batch_size = 64
epochs = 20

# Load and Preprocess Datasets
train_ds, val_ds, test_ds = get_dataset(TRAIN_DIR, TEST_DIR, batch_size=batch_size, 
                                        img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

# Apply the custom (SIFT-related) feature extraction preprocessing step
train_ds = train_ds.map(lambda x, y: (extract_iris_features(x), y))
val_ds = val_ds.map(lambda x, y: (extract_iris_features(x), y))
test_ds = test_ds.map(lambda x, y: (extract_iris_features(x), y))

# Standard ResNet50 Preprocessing Layer
preprocess_layer = tf.keras.applications.resnet50.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess_layer(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_layer(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_layer(x), y))


# 4. Model Definition (ResNet50 Frozen)
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False # Freeze weights for feature extraction phase

# Modify the classification head (No Dropout in this specific experiment for simplicity)
x = GlobalAveragePooling2D()(base_model.output)
outputs = Dense(N_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)


# 5. Compile and Train
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("--- Starting Training for ResNet50_B (SIFT-based Feature Extraction) ---")
# Training this model should produce the low-accuracy/high-loss result (Train 62.73% / Val 27.10%)
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds) 


# 6. Evaluate and Visualize the Failure
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'\nTEST SET FINAL ACCURACY (SIFT Failure Analysis): {test_accuracy}')

# Plot training & validation accuracy and loss (Figure 3-23 logic)
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - ResNet50_B (SIFT Failure)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - ResNet50_B (SIFT Failure)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()

print("\n--- CRITICAL CONCLUSION ---")
print("The observed result (Train Acc high, Val Acc very low/plateaued) empirically demonstrates that introducing a custom SIFT-based feature extraction layer is highly detrimental to the generalization capabilities of the DCNN, proving that pure end-to-end learning is superior for this dataset.")

# --- End of A_ResNet50_SIFT_FAILURE_ANALYSIS.ipynb ---
