# --- MobileNetV2_Edge_Architecture.py ---
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense

# --- Parameters ---
IMG_SHAPE = (224, 224, 3) # Standard input size
N_CLASSES = 50            # Total number of classes
DROPOUT_RATE = 0.5        # Your chosen regularization rate

def build_mobilenet_model(input_shape, num_classes, dropout_rate):
    """
    Builds the MobileNetV2-based DCNN model using ImageNet Transfer Learning.
    This architecture is preferred for its lightweight size and efficiency on edge devices.
    """
    # 1. Base Model (MobileNetV2 pre-trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=input_shape, 
        include_top=False,        # Remove the original classification head
        weights='imagenet'        # Use pre-trained weights
    )

    # 2. Freeze the base model's weights
    base_model.trainable = False

    # 3. Define the new Classification Head
    inputs = Input(shape=input_shape)
    
    # Pre-process inputs (standard MobileNet pre-processing)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
    
    # Pass inputs through the frozen base model
    x = base_model(x, training=False) 
    
    # Add a global layer (from your architecture)
    x = GlobalAveragePooling2D()(x)
    
    # Add Dropout for regularization
    x = Dropout(dropout_rate)(x) 
    
    # Final Classification Layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs, outputs)
    
    return model

# --- Model Instantiation and Summary ---
mobilenet_model = build_mobilenet_model(IMG_SHAPE, N_CLASSES, DROPOUT_RATE)

print("--- MobileNetV2 Model Architecture Summary (Frozen Base) ---")
mobilenet_model.summary()

# Verify the number of trainable parameters
trainable_params = sum([tf.keras.backend.count_params(p) for p in mobilenet_model.trainable_weights])
print(f"\nTotal Trainable Parameters: {trainable_params}")
print(f"Observation: MobileNetV2 is significantly lighter than ResNet50 (fewer total parameters), ideal for edge deployment.")

# --- End of MobileNetV2_Edge_Architecture.py ---
