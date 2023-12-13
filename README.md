## Green-Guard

## Overview
This repository contains a robust set of data pre-processing and visualization steps to prepare and understand a dataset of plant images, A notable dataset features 87,000 RGB images across 38 distinct classes. The primary objective of this research endeavor is to design a classifier proficient in discerning a plant's health condition based on an image. Examples of classes include "Strawberry - Healthy", "Tomato - Late Blight", "Apple - Cedar Apple Rust", and "Soybean - Healthy”.


## Table of Contents
- [Data Collection and Preprocessing](#datacollecion)
    - Data Collection: Collect images of different plants and their associated disease types.
    - Data Analysis: Analyze the data completely.
    -  Data Labeling: Create a Data Frame with Image Paths and Labels.
    -  Data Splitting: Split the Dataset for Training and Testing.
- [Exploratory Data Analysis (EDA)](#EDA)
    - Visualization of Diseases by Plant.
    - Find value counts for each disease type with the Plant and Visualize it.
    - Find Value counts for each plant type and Visualize it.
- [Data Preparation and Augmentation](#datapreparation)
    - Image Encoding: Convert Images to Numericals using Encoders.
    - Data Augmentation and Batching: Create a TensorFlow data generator.
- [Model Building](#modelbuild)
    - Transfer Learning: Setup model using InceptionV3 with custom layers.
    - Custom Model: Build another model tailored to the project.
- [Model Evaluation](#modelevaluation)
    - Evaluate Transfer Learing model.
    - Evaluate Custom Model.
- [Model Deployment](#modeldeploy)
    - Deploy the model for real-time predictions. a
    


# Plant Disease Classification: Data Pre-Processing

- ## Visualization using seaborn based on the sorted data
  
  ![Unknown-13](https://github.com/harishmanyam9/Green-Guard/assets/113054457/a75221b7-dc86-4a43-acbe-ef3f752a5ddc)

  

- ## Value Counts of Diseases for Each Plant
  
  ![Unknown-14](https://github.com/harishmanyam9/Green-Guard/assets/113054457/daecacdd-5d65-4e39-955f-6a3aeffa7946)

  

- ## Plot disease counts for plants
  
  ![Unknown-15](https://github.com/harishmanyam9/Green-Guard/assets/113054457/a909172b-4280-4142-ac61-52aa969ccae9)

# Plant Disease Classification: Data Preprocessing and Model Preparation

## Data Splitting for Training and Testing

Before training the model, it's crucial to split the dataset into training and testing sets. This approach helps in evaluating the model's performance objectively.

### Splitting the Dataset
We use `train_test_split` from `sklearn.model_selection` to divide the dataset. This function shuffles the dataset and splits it into training and testing sets. The `test_size` parameter determines the proportion of the dataset to include in the test split.

- `test_size=0.33` means that 33% of the dataset will be used for testing.
- `stratify=result['combined']` ensures that the split is done in a stratified fashion, using the labels. This makes sure that both training and testing sets have a similar distribution of each class.

```python
from sklearn.model_selection import train_test_split

# Splitting the dataset into Training and Testing sets
Train, Test = train_test_split(result, test_size=0.33, stratify=result['combined'])
```



# Plant Disease Classification: Model Building

## Building the Convolutional Neural Network

In this section, we construct the Convolutional Neural Network (CNN) for plant disease classification. Our approach utilizes the InceptionV3 architecture, a powerful pre-trained model, and adds custom layers to tailor it for our specific task.

### Setting Up the Model
1. **Input Layer**: Define the input shape as \(400 \times 400\) pixels with 3 channels (RGB).
2. **Base Model**: Load the pre-trained InceptionV3 model, keeping its original weights (trained on ImageNet) and excluding its top layer.
3. **Custom Convolutional Layers**: Add additional convolutional layers to further process the features extracted by InceptionV3.
4. **Global Average Pooling**: Implement Global Average Pooling to reduce the spatial dimensions.
5. **Fully Connected Layers**: Add dense layers for deeper feature learning, including Batch Normalization and Dropout for regularization.
6. **Output Layer**: A softmax layer to classify the images into 120 different classes representing various plant diseases.

### Model Architecture Code
```python
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2

# Define input shape and layer
input_shape = (400, 400, 3)
input_layer = Input(shape=input_shape)

# Load pre-trained InceptionV3 and set up custom layers
base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in base_inception.layers:
    layer.trainable = True

x = base_inception(input_layer)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
# ... [Additional layers are added here] ...
output_layer = Dense(120, activation='softmax')(x)

# Construct the model
Inception = Model(inputs=input_layer, outputs=output_layer)
```

## Model Training Callbacks

To enhance the training process and ensure optimal performance, we use several callback functions in our model:

1. **EarlyStopping**: Monitors the validation loss and stops training if it doesn't decrease after three epochs. This helps in preventing overfitting.
   
2. **ModelCheckpoint**: Saves the best model based on the minimum validation loss. This ensures that we always have the best performing model saved during training.
   
3. **ReduceLROnPlateau**: Reduces the learning rate when the validation loss plateaus, improving training efficiency and performance.

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=3)
mc = ModelCheckpoint('/content/drive/MyDrive/Capstone_1/Inception.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, verbose=0)

```










