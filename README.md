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
    - Deploy the model for real-time predictions.
    


# Plant Disease Classification: Data Pre-Processing

- ## Visualization using seaborn based on the sorted data
  
  ![Unknown-13](https://github.com/harishmanyam9/Green-Guard/assets/113054457/a75221b7-dc86-4a43-acbe-ef3f752a5ddc)

  

- ## Value Counts of Diseases for Each Plant
  
  ![Unknown-14](https://github.com/harishmanyam9/Green-Guard/assets/113054457/daecacdd-5d65-4e39-955f-6a3aeffa7946)

  

- ## Plot disease counts for plants
  
  ![Unknown-15](https://github.com/harishmanyam9/Green-Guard/assets/113054457/a909172b-4280-4142-ac61-52aa969ccae9)






