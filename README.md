# Fast Food Classification - Neural Networks

## Project Overview
Addressing the problem of fast food classification, where class represent types of fast food items and input data consist of
images. The objective is to train a neural network to correctly classify a given unknown image into the appropriate category

## Dataset & Preprocessing
- Data split into **Training, Validation, and Test sets**.
- **Balanced training & validation sets**, while the test set is imbalanced.
- **Data Augmentation:** random flipping, rotation, brightness, and contrast adjustments.
- **Scaling**: Ensures pixel values are within a suitable range for the model

## Neural Network Model
- **Loss Function:** Sparse Categorical Crossentropy (used for multi-class classification).
- **Activation Function:** ReLU (addresses vanishing gradients, with potential Leaky ReLU usage).
- **Optimizer:** Adam (efficient and widely used for deep learning)

## Architecture
- Conv2D layers
- MaxPooling layers 
- Dropout layer (30%) 
- Flatten & Dense layers

## Technologies Used
- Python  
