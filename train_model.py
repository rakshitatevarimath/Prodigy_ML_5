import os
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. START MESSAGE
# ============================================================

print("=" * 50)
print("FOOD RECOGNITION TRAINING STARTED")
print("=" * 50)

print("TensorFlow version:", tf.__version__)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 10

TRAIN_PATH = "dataset5/data/food-101-tiny/train"
VALID_PATH = "dataset5/data/food-101-tiny/valid"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "food_model.keras"
)

CLASS_NAMES_PATH = "class_names.json"


# ============================================================
# 3. CHECK DATASET FOLDERS
# ============================================================

print("\nChecking dataset folders...")

if not os.path.exists(TRAIN_PATH):
    print("ERROR: Training folder not found!")
    print("Expected:", TRAIN_PATH)
    exit()

if not os.path.exists(VALID_PATH):
    print("ERROR: Validation folder not found!")
    print("Expected:", VALID_PATH)
    exit()

print("Training folder found:", TRAIN_PATH)
print("Validation folder found:", VALID_PATH)


# ============================================================
# 4. CREATE MODELS FOLDER
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 5. IMAGE DATA GENERATORS
# ============================================================

print("\nPreparing image data...")


# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode="nearest"
)


# Validation data
# No augmentation for validation images
valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================================
# 6. LOAD TRAINING DATA
# ============================================================

print("\nLoading training images...")

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True
)


# ============================================================
# 7. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation images...")

validation_data = valid_datagen.flow_from_directory(
    VALID_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# 8. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("Training images:", train_data.samples)

print("Validation images:", validation_data.samples)

print("Number of classes:", train_data.num_classes)

print("\nClass names and numbers:")

print(train_data.class_indices)


# ============================================================
# 9. SAVE CLASS NAMES
# ============================================================

class_names = list(train_data.class_indices.keys())

with open(CLASS_NAMES_PATH, "w") as file:

    json.dump(class_names, file, indent=4)


print("\nClass names saved to:", CLASS_NAMES_PATH)

print("\nClasses:")

for number, name in enumerate(class_names):

    print(number, "->", name)


# ============================================================
# 10. LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)
)


# ============================================================
# 11. FREEZE PRETRAINED LAYERS
# ============================================================

base_model.trainable = False


# ============================================================
# 12. CREATE NEW CLASSIFICATION LAYERS
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(
    0.5
)(x)


output = Dense(
    train_data.num_classes,

    activation="softmax"
)(x)


# ============================================================
# 13. CREATE FINAL MODEL
# ============================================================

model = Model(
    inputs=base_model.input,

    outputs=output
)


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 15. DISPLAY MODEL
# ============================================================

print("\n" + "=" * 50)
print("MODEL CREATED")
print("=" * 50)

model.summary()


# ============================================================
# 16. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n" + "=" * 50)
print("STARTING TRAINING")
print("=" * 50)

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# 18. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 50)
print("EVALUATING MODEL")
print("=" * 50)

loss, accuracy = model.evaluate(
    validation_data
)

print(
    f"Validation Loss: {loss:.4f}"
)

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# 20. FINISHED
# ============================================================

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("Model saved at:")

print(MODEL_PATH)

print("\nClass names saved at:")

print(CLASS_NAMES_PATH)

print("\nYour model is ready for prediction!")