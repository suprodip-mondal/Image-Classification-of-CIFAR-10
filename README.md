# Image-Classification-of-CIFAR-10

PROJECT OVERVIEW
This project implements a deep learning pipeline to classify images from the CIFAR-10 dataset into ten categories[cite: 1, 23]. The model uses a custom Convolutional Neural Network (CNN) architecture built with PyTorch.

DATASET DETAILS
- Dataset: CIFAR-10
- Image Size: 3x32x32 pixels (RGB)
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck 
- Training Samples: 50,000 
- Test Samples: 10,000

MODEL ARCHITECTURE
The model consists of two main parts:
1. CNN Feature Extractor:
   - Layer 1: Conv2d(3 to 16 filters) + ReLU + MaxPool
   - Layer 2: Conv2d(16 to 32 filters) + ReLU + MaxPool
   - Layer 3: Conv2d(32 to 64 filters) + ReLU + MaxPool
2. DeepNN Classifier:
   - Linear layer (1024 to 512)
   - Linear layer (512 to 128) 
   - Output layer (128 to 10) 

Total Parameters: 615,338 

TRAINING AND PERFORMANCE
- Optimizer: Adam (Learning Rate: 0.0005)
- Loss Function: CrossEntropyLoss 
- Epochs: 15


Requirements:
torch
torchvision
numpy
pandas
matplotlib
seaborn
torchsummary
scikit-learn

- Final Test Accuracy: 73% 

