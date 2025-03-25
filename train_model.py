import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import shutil

# Define data directories
DATA_DIR = 'EYE_sample'
TRAIN_DIR = 'data/train_eyes'
VAL_DIR = 'data/valid_eyes'
IMG_WIDTH, IMG_HEIGHT = 24, 24
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
MODEL_PATH = 'models/eye_state_model.h5'

# Create train and validation directories if they don't exist
os.makedirs(os.path.join(TRAIN_DIR, 'open'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'closed'), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR, 'open'), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR, 'closed'), exist_ok=True)

# Organize the downloaded data into train and validation sets (Run only once or when new data is added)
if not os.listdir(os.path.join(TRAIN_DIR, 'open')) or not os.listdir(os.path.join(TRAIN_DIR, 'closed')) or \
   not os.listdir(os.path.join(VAL_DIR, 'open')) or not os.listdir(os.path.join(VAL_DIR, 'closed')):
    print("Organizing data...")
    for subdir, _, files in os.walk(DATA_DIR):
        category = os.path.basename(subdir)
        if category == 'close_look':
            label = 'closed'
        elif category in ['left_look', 'right_look', 'forward_look']:
            label = 'open'
        else:
            continue  # Skip other directories

        # Shuffle files for random splitting
        np.random.shuffle(files)
        split_index = int(len(files) * (1 - VALIDATION_SPLIT))
        train_files = files[:split_index]
        val_files = files[split_index:]

        for filename in train_files:
            src = os.path.join(subdir, filename)
            dst = os.path.join(TRAIN_DIR, label, filename)
            shutil.copy2(src, dst)

        for filename in val_files:
            src = os.path.join(subdir, filename)
            dst = os.path.join(VAL_DIR, label, filename)
            shutil.copy2(src, dst)
    print("Data organized.")
else:
    print("Training and validation data directories already exist. Skipping data organization.")

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
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary'
)

# Define the CNN model architecture
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
    Dense(1, activation='sigmoid')  # Binary classification: open or closed
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Save the trained model
model.save('models/eye_state_model_new_data.h5', overwrite=True)
print("Trained eye state model on new data saved!")