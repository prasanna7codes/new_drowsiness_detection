import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import precision_score, recall_score, f1_score

# Define data directories
DATA_DIR = 'EYE_sample'
TRAIN_DIR = 'data/train_eyes'
VAL_DIR = 'data/valid_eyes'
IMG_WIDTH, IMG_HEIGHT = 24, 24
BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = 'models/eye_state_model.h5'

# Ensure directories exist
os.makedirs(os.path.join(TRAIN_DIR, 'open'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'closed'), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR, 'open'), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR, 'closed'), exist_ok=True)

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False
)

# Display dataset information
print(f"Total training images (before augmentation): {train_generator.samples}")
steps_per_epoch = train_generator.samples // BATCH_SIZE
augmented_images_per_epoch = steps_per_epoch * BATCH_SIZE
print(f"Images processed per epoch (augmented + original): {augmented_images_per_epoch}")
total_augmented_images = augmented_images_per_epoch * EPOCHS
print(f"Total augmented images across all epochs: {total_augmented_images}")

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model with additional metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save the model
model.save(MODEL_PATH)
print("Model training completed and saved!")

# Evaluate the model
val_preds = model.predict(validation_generator)
y_true = validation_generator.classes
y_pred = (val_preds > 0.5).astype(int).flatten()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-Score: {f1:.4f}")
