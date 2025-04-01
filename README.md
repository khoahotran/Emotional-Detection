# Emotion Recognition Model

This project implements a Convolutional Neural Network (CNN) to recognize human emotions from facial images. The model is trained on a dataset of grayscale images categorized into seven emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. TensorFlow and Keras are used to build, train, and evaluate the model, which can predict emotions from new images.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)
- [License](#license)

## Overview

This project aims to classify facial expressions into one of seven emotion categories. It processes 48x48 grayscale images and outputs a predicted emotion label. The system includes data preprocessing, model training, and prediction functionalities for classifying new images.

## Dataset

The dataset should be structured as follows:

- `images/train/`: Training images, split into subdirectories for each emotion (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
- `images/validation/`: Validation images, similarly organized.

Each subdirectory contains grayscale images of faces labeled with the corresponding emotion. The dataset is loaded into Pandas DataFrames and preprocessed into NumPy arrays for training.

## Requirements

Ensure that the following dependencies are installed:

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TQDM
- Pickle (for data serialization)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/khoahotran/Emotional-Detection.git
   cd Emotional-Detection
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow>=2.0 numpy pandas matplotlib scikit-learn tqdm
   ```

## Usage

### Data Preprocessing

[Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset is processed and saved as NumPy arrays or a Pickle file:


```python
train = createdataframe('images/train')
test = createdataframe('images/validation')
x_train, x_test = extract_features(train['image']), extract_features(test['image'])
y_train, y_test = to_categorical(le.fit_transform(train['label']), num_classes=7), to_categorical(le.transform(test['label']), num_classes=7)



# Save preprocessed data
np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Alternatively, use Pickle:
with open("train_test_data.pkl", "wb") as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
```



### Training the Model

Train the model using the preprocessed data:

```python
model = Sequential([...])  # See Model Architecture section
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Load preprocessed data
x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=40, validation_data=(x_test, y_test))

# Save the trained model
model.save("trained.keras")
```

### Making Predictions

Use the trained model to predict emotions from new images:

```python
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    img = np.array(img).reshape(1, 48, 48, 1) / 255.0
    pred_label = label[np.argmax(model.predict(img))]
    plt.imshow(img.reshape(48, 48), cmap='gray')
    plt.title(f"Predicted: {pred_label}")
    plt.show()

# Example usage
predict_emotion('images/validation/sad/505.jpg')
```

## Model Architecture

The CNN model consists of the following layers:

1. **Conv2D**: 64 filters, 3x3 kernel, ReLU activation, input shape (48, 48, 1)
2. **BatchNormalization**
3. **MaxPooling2D**: 2x2 pool size
4. **Dropout**: 0.3
5. **Conv2D**: 128 filters, 3x3 kernel, ReLU activation
6. **BatchNormalization**
7. **MaxPooling2D**: 2x2 pool size
8. **Dropout**: 0.3
9. **Conv2D**: 256 filters, 3x3 kernel, ReLU activation
10. **BatchNormalization**
11. **MaxPooling2D**: 2x2 pool size
12. **Dropout**: 0.4
13. **Flatten**
14. **Dense**: 512 units, ReLU activation
15. **Dropout**: 0.4
16. **Dense**: 7 units, Softmax activation (output layer)

The model is compiled with the Adam optimizer and categorical cross-entropy loss.

## File Structure

The project directory contains the following files:

```
emotion-recognition-model/
├── images/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   └── ...
│   └── validation/
│       ├── angry/
│       ├── disgust/
│       └── ...
├── trained.keras            # Trained model file
├── x_train.npy              # Preprocessed training images
├── x_test.npy               # Preprocessed validation images
├── y_train.npy              # Preprocessed training labels
├── y_test.npy               # Preprocessed validation labels
├── train_test_data.pkl      # Alternative preprocessed data file
├── emotion.ipynb            # Jupyter notebook with code
└── README.md                # This file
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
