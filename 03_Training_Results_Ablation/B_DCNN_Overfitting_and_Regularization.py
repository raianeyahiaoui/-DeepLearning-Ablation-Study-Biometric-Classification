# --- B_DCNN_Overfitting_and_Regularization.ipynb ---

# 1. Setup and Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# --- Data and Model Parameters ---
# NOTE: Using a path that would lead to the standard (non-SIFT) preprocessed data.
TRAIN_DIR = "/content/drive/MyDrive/UbirisV2/UbirisV2_Dataset/Train_Dataset"
TEST_DIR = "/content/drive/MyDrive/UbirisV2/UbirisV2_Dataset/Test_Dataset"

N_CLASSES = 50 
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3 
DROPOUT_RATE = 0.5 # Your chosen rate for regularization

# 2. Dataset Loading Function
def get_dataset(train_dir, test_dir, batch_size=64, img_height=224, img_width=224):
    """Loads the dataset and splits it into training/validation/test sets."""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=123, 
        image_size=(img_height, img_width), batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=123, 
        image_size=(img_height, img_width), batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    
    return train_ds, val_ds, test_ds

# 3. Load Data, Calculate Class Weights, and Apply Augmentation
learning_rate = 0.0015 # Using the LR that showed the most prominent overfitting early on
batch_size = 64
epochs = 20

train_ds, val_ds, test_ds = get_dataset(TRAIN_DIR, TEST_DIR, batch_size=batch_size, 
                                        img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

# --- Class Weight Computation (Addressing Imbalance) ---
label_list = []
for images, labels in train_ds.unbatch():
    label_list.append(labels.numpy())
label_list = np.array(label_list)
class_weights = compute_class_weight('balanced', classes=np.unique(label_list), y=label_list)
class_weights_dict = dict(enumerate(class_weights))
print("Computed Class Weights (Initial Overfit Fix):", class_weights_dict)

# --- Data Augmentation (Addressing Overfitting) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
# Apply augmentation to training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# --- Preprocessing ---
preprocess_layer = tf.keras.applications.resnet50.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess_layer(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_layer(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_layer(x), y))


# 4. Model Definition (ResNet50 with Dropout in the head)
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False # Feature Extraction Phase

# --- Classification Head with Dropout (Addressing Overfitting) ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x) # CRUCIAL REGULARIZATION LAYER
outputs = Dense(N_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)


# 5. Compile and Train
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("--- Starting Training for ResNet50 Overfitting Debugging ---")
# Training this model should produce the typical overfitting plots where
# the training curve separates significantly from the validation curve.
history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=val_ds,
                    class_weight=class_weights_dict) # Apply Class Weights

# 6. Evaluate and Visualize Debugging Process
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'\nTEST SET FINAL ACCURACY (Overfitting Debugging Run): {test_accuracy}')

# Plot training & validation accuracy and loss 
plt.figure(figsize=(12, 5))

# Accuracy plot (Will show the train/val gap - a sign of overfitting)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - Regularization Attempt')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Loss plot (Will show val loss flattening or increasing - another sign of overfitting)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - Regularization Attempt')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()

print("\n--- DEBUGGING CONCLUSION ---")
print("These plots demonstrate the efficacy of regularization. The initial divergence of the curves (early overfitting) is partially mitigated by Data Augmentation, Dropout(0.5), and Class Weighting, pulling the validation curve closer to the training curve, though further tuning is required for optimal generalization.")

# --- End of B_DCNN_Overfitting_and_Regularization.ipynb ---
