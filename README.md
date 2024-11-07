# Defect Detection in Product Quality Inspection  

Description:  

This repository implements a convolutional neural network (CNN) model using TensorFlow to detect defects in product images for quality control applications. The model is trained on the NEU Metal Surface Defects Dataset, which includes various defect types such as Scratches, Pitted, Rolled, Patches, Inclusion, and Crazing.  

Prerequisites:  

Python 3.x   
TensorFlow  
NumPy   
Pandas  
OpenCV (cv2)  
Matplotlib  
SciPy  

Installation:  

Create a virtual environment (recommended).  
Install the required libraries using pip install numpy pandas opencv-python matplotlib scipy tensorflow.  

Data Preparation (Modify for your data):  

Download the NEU Metal Surface Defects Dataset or use your own dataset with the same directory structure (train, validation, test subdirectories containing class folders).  
Ensure your data structure matches the training and validation directory paths in the code (train_dir, val_dir, test_dir).

Model Architecture:  

The model architecture consists of:  

Convolutional Layers: Extract features from the image using multiple convolutional layers with ReLU activation.  
Max Pooling Layers: Reduce image size and capture spatial relationships between features.  
Flatten Layer: Convert the 2D feature maps into a 1D vector for the dense layers. 
Dense Layers: Learn complex relationships between the extracted features and the defect classes.  
Dropout Layer: Prevent overfitting by randomly dropping out a portion of neurons during training.  
Output Layer: Categorical output layer with sigmoid activation to predict the presence/absence of each defect type (6 classes in this example).  

Evaluation:  
 
The script trains the model for 20 epochs with early stopping if the validation accuracy reaches 92% (customizable). It then evaluates the final model's accuracy and loss on both the training and validation sets.  
