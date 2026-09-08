import tensorflow as tf
from tensorflow.keras import layers, models, activations

def build_eye_cnn(input_shape=(24, 24, 1)):
    """
    Builds a simple CNN model for eye state feature extraction.
    
    Args:
        input_shape (tuple): The shape of the input eye images.
        
    Returns:
        A TensorFlow Keras model.
    """
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid')) # Output layer for open/closed classification
    
    return model

def get_feature_extractor_model(base_model):
    """
    Returns a model that extracts features from the second to last dense layer.
    """
    # The feature vector will be the output of the last dense layer before the classification head
    feature_extractor = models.Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-3].output, # Output of the 'Dense' layer
    )
    return feature_extractor

if __name__ == '__main__':
    # Example of how to build and summarize the models
    
    # Build the classification model
    eye_classification_model = build_eye_cnn()
    print("--- Eye Classification Model Summary ---")
    eye_classification_model.summary()
    
    # Build the feature extractor model
    eye_feature_extractor = get_feature_extractor_model(eye_classification_model)
    print("\n--- Eye Feature Extractor Model Summary ---")
    eye_feature_extractor.summary()
    
    # Example of using the feature extractor
    # Create a dummy eye image
    dummy_eye = tf.random.normal([1, 24, 24, 1])
    features = eye_feature_extractor(dummy_eye)
    print(f"\nOutput feature vector shape: {features.shape}")
