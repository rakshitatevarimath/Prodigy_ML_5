# Food Recognition and Calorie Estimation Using Deep Learning

## 📌 Project Overview

This project is developed as part of **Prodigy Infotech Task-05**.

The objective is to develop a deep learning model that can recognize food items from images and provide an approximate calorie estimate for the predicted food item.

The project uses the **Food-101-Tiny** dataset and **MobileNetV2 Transfer Learning** for food image classification.

---

## 🎯 Objective

The main objectives of this project are:

* Recognize food items from images.
* Classify an input image into a food category.
* Display the model's prediction confidence.
* Estimate calories using a calorie reference table.
* Demonstrate the use of deep learning for food recognition.

---

## 🛠️ Technologies Used

* Python 3.12
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* Scikit-learn
* VS Code
* GitHub

---

## 📂 Dataset

The project uses the **Food-101-Tiny** dataset, which is a smaller dataset based on Food-101 and is suitable for experimentation and model development.

The dataset is organized into:

```text
train/
valid/
```

The training and validation folders contain images arranged according to their food classes.

---

## 🧠 Model

This project uses **MobileNetV2** with transfer learning.

### Why MobileNetV2?

MobileNetV2 is a lightweight convolutional neural network that is suitable for image classification and can be used efficiently on computers with limited resources.

The pretrained ImageNet layers are used as the base model, and additional classification layers are added for the food categories.

### Model Architecture

```text
Input Image
     ↓
Image Resizing (224 × 224)
     ↓
Normalization
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Dropout
     ↓
Softmax Classification
     ↓
Predicted Food
```

---

## 🍽️ Food Recognition

The trained model receives an image as input and predicts the most likely food category.

Example:

```text
Input:
Food Image

Output:
Food: Sushi
Confidence: 85.32%
```

The exact prediction and confidence depend on the input image and trained model.

---

## 🔥 Calorie Estimation

After identifying the food item, the project uses a calorie reference dictionary to provide an approximate calorie value.

Example:

```text
Food: Sushi
Estimated Calories: 250 kcal
```

The calorie values are approximate reference values. Actual calories can vary depending on portion size, ingredients, preparation method, and serving size.

---

## 📁 Project Structure

```text
PRODIGY_ML_5/
│
├── dataset5/
│   └── data/
│       └── food-101-tiny/
│           ├── train/
│           └── valid/
│
├── models/
│   └── food_model.keras
│
├── test_images/
│   └── food.jpg
│
├── calorie_data.py
├── class_names.json
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the project

```bash
cd PRODIGY_ML_5
```

### 3. Create a virtual environment

```bash
py -3.12 -m venv venv
```

### 4. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run:

```bash
python train_model.py
```

The training script loads the training and validation images, trains the MobileNetV2-based model, and saves the trained model inside the `models` folder.

The trained model is saved as:

```text
models/food_model.keras
```

---

## 🔍 Make a Prediction

Place a test food image inside:

```text
test_images/
```

For example:

```text
test_images/food.jpg
```

Then run:

```bash
python predict.py
```

The program displays:

```text
Food Recognition Result

Food: <predicted food>
Confidence: <confidence percentage>
Estimated Calories: <approximate calories> kcal
```

---

## 📊 Example Output

```text
=============================================
       FOOD RECOGNITION RESULT
=============================================
Food: Sushi
Confidence: 85.32%
Estimated Calories: 250 kcal
=============================================
```

The displayed result will vary depending on the test image.

---

## 🚀 Future Improvements

The project can be improved by:

* Using the complete Food-101 dataset.
* Increasing the number of training images.
* Fine-tuning MobileNetV2 layers.
* Adding top-3 food predictions.
* Creating a Flask web application.
* Adding an image upload interface.
* Using portion-size estimation for more accurate calorie estimation.
* Adding a daily food intake tracking system.
* Improving calorie information using a more detailed nutritional database.

---

## 📚 Learning Outcomes

Through this project, I learned about:

* Image classification.
* Convolutional Neural Networks.
* Transfer learning.
* MobileNetV2.
* Image preprocessing.
* Data augmentation.
* Model training and validation.
* Model prediction.
* Python and TensorFlow.
* Basic calorie estimation using a reference dataset.

---

## 👩‍💻 Internship Task

**Organization:** Prodigy Infotech

**Task:** Task-05

**Project:** Food Recognition and Calorie Estimation Using Deep Learning

**Track:** Machine Learning
