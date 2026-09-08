import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from part_a.cnn_model import build_eye_cnn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --- Configuration ---
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../../drowsiness_dataset')
IMG_WIDTH, IMG_HEIGHT = 24, 24
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '../../models/eye_cnn_weights.h5')
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15 # of the training set

def train_model():
    """
    Trains the eye-tracking CNN model.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at '{DATASET_DIR}'")
        print("Please download the 'Drowsiness dataset' from Kaggle and place it in the project root.")
        return

    # --- Load Image Paths and Labels ---
    image_paths = []
    labels = []

    drowsy_dir = os.path.join(DATASET_DIR, 'Drowsy')
    non_drowsy_dir = os.path.join(DATASET_DIR, 'Non Drowsy')

    if not os.path.exists(drowsy_dir) or not os.path.exists(non_drowsy_dir):
        print(f"Error: Expected 'Drowsy' and 'Non Drowsy' subdirectories in '{DATASET_DIR}'.")
        print("Please ensure the dataset structure is correct.")
        return

    for img_name in os.listdir(drowsy_dir):
        image_paths.append(os.path.join(drowsy_dir, img_name))
        labels.append(0) # 0 for Drowsy

    for img_name in os.listdir(non_drowsy_dir):
        image_paths.append(os.path.join(non_drowsy_dir, img_name))
        labels.append(1) # 1 for Non Drowsy

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Shuffle the data
    image_paths, labels = shuffle(image_paths, labels, random_state=42)

    # --- Split Data into Train, Validation, Test ---
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, random_state=42, stratify=y_train
    )

    print(f"Total images: {len(image_paths)}")
    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Test images: {len(X_test)}")

    # --- Custom Data Generators ---
    def data_generator(img_paths, lbls, batch_size, augment=False):
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2 if augment else 0,
            zoom_range=0.2 if augment else 0,
            horizontal_flip=True if augment else False
        )
        
        # Create a flow from numpy arrays (image data and labels)
        # We need to load images first
        images = []
        for img_path in img_paths:
            img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
            img = img_to_array(img)
            images.append(img)
        images = np.array(images)

        return datagen.flow(images, lbls, batch_size=batch_size)

    train_generator = data_generator(X_train, y_train, BATCH_SIZE, augment=True)
    validation_generator = data_generator(X_val, y_val, BATCH_SIZE)
    test_generator = data_generator(X_test, y_test, BATCH_SIZE)

    # --- Model Building ---
    model = build_eye_cnn(input_shape=(IMG_WIDTH, IMG_HEIGHT, 1))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n--- Model Summary ---")
    model.summary()

    # --- Model Training ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(X_val) // BATCH_SIZE
    )

    # --- Model Evaluation ---
    print("\n--- Evaluating Model on Test Set ---")
    test_loss, test_acc = model.evaluate(test_generator, steps=len(X_test) // BATCH_SIZE)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # --- Save the Trained Model ---
    print(f"\n--- Saving Trained Model to '{MODEL_SAVE_PATH}' ---")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    train_model()

