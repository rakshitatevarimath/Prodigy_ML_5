import json
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from calorie_data import CALORIES


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/food_model.keras"

CLASS_NAMES_PATH = "class_names.json"

IMAGE_PATH = "test_images/food.jpg"


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found!")
    print("Expected:", MODEL_PATH)
    exit()

if not os.path.exists(CLASS_NAMES_PATH):
    print("ERROR: class_names.json not found!")
    exit()

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Test image not found!")
    print("Expected:", IMAGE_PATH)
    exit()


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


print("\nFood classes:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ==========================================
# LOAD IMAGE
# ==========================================

print("\nLoading test image...")

img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)


# ==========================================
# PREPROCESS IMAGE
# ==========================================

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0


# ==========================================
# PREDICTION
# ==========================================

print("Predicting food...")

predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])

predicted_food = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100


# ==========================================
# CALORIE ESTIMATION
# ==========================================

calories = CALORIES.get(
    predicted_food,
    "Not available"
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 45)

print("       FOOD RECOGNITION RESULT")

print("=" * 45)

print("Food:", predicted_food)

print(f"Confidence: {confidence:.2f}%")

print("Estimated Calories:", calories, "kcal")

print("=" * 45)