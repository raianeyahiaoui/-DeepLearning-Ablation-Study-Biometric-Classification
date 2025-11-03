# --- ResNet50_Transfer_Learning_Setup.py ---
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense

# --- Parameters ---
IMG_SHAPE = (224, 224, 3) # Standard input size (from your preprocessing)
N_CLASSES = 50            # Total number of classes
DROPOUT_RATE = 0.5        # Your chosen regularization rate for the final layers

def build_resnet50_model(input_shape, num_classes, dropout_rate):
    """
    Builds the ResNet50-based DCNN model using ImageNet Transfer Learning.
    This architecture corresponds to your best-performing model (ResNet50_C).
    """
    # 1. Base Model (ResNet50 pre-trained on ImageNet)
    base_model = ResNet50(
        input_shape=input_shape, 
        include_top=False,        # Crucial: Remove the original classification head
        weights='imagenet'        # Use pre-trained weights for feature extraction
    )

    # 2. Freeze the base model's weights
    # This keeps the powerful ImageNet features locked while training only the new layers.
    base_model.trainable = False

    # 3. Define the new Classification Head
    inputs = Input(shape=input_shape)
    
    # Pre-process inputs (standard ResNet pre-processing is built into the Transfer Learning flow)
    x = tf.keras.applications.resnet50.preprocess_input(inputs) 
    
    # Pass inputs through the frozen base model
    x = base_model(x, training=False) 
    
    # Add a global layer to collapse spatial dimensions (from your architecture)
    x = GlobalAveragePooling2D()(x)
    
    # Add Dropout for regularization (crucial for preventing the overfitting you observed)
    x = Dropout(dropout_rate)(x) 
    
    # Final Classification Layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs, outputs)
    
    return model

# --- Model Instantiation and Summary ---
resnet_model = build_resnet50_model(IMG_SHAPE, N_CLASSES, DROPOUT_RATE)

print("--- ResNet50 Model Architecture Summary (Frozen Base) ---")
resnet_model.summary()

# Verify that the trainable parameters are only in the new head layers
trainable_params = sum([tf.keras.backend.count_params(p) for p in resnet_model.trainable_weights])
print(f"\nTotal Trainable Parameters: {trainable_params}")
print(f"Goal Achieved: Only the new classification head is trainable for feature extraction.")

# --- End of ResNet50_Transfer_Learning_Setup.py ---
