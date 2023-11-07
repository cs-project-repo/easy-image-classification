Easy Image Classifier

Introduction

The Easy Image Classifier is a Python library that simplifies the process of training and using image classification models. It provides a straightforward way to train a deep learning model for image classification and to make predictions on new images using the trained model. 

Installation

You can use the Easy Image Classifier by cloning the GitHub repository and importing the EasyImageClassifier class into your project.

```
from easy_image_classifier import EasyImageClassifier
```

Usage

To get started with the Easy Image Classifier, I've prepared a step-by-step guide in a Google Colab notebook. This notebook demonstrates how to train image classification models and make predictions using the Easy Image Classifier library. Follow this link to access the Google Colab notebook: [Google Colab](https://drive.google.com/file/d/1e7R_e-sF01YC2-h4elJvUpqANEoXTlJb/view?usp=sharing)

Params

Here are the parameters you can customize when using the Easy Image Classifier:

| params     | type       | default   | notes                                      |
| :------------- | :---------- | :--------- | :------------------------------------------ |
| data          | string       | '~/easy_image_classifier/data' | Path to the directory containing the image dataset.* (Read below) |
| model_type    | string        | 'resnet101' | Type of pre-trained model to use (e.g., 'resnet50'). |
| num_epochs    | number      | 25        | Number of training epochs.                |
| lr            | float      | 0.001     | Learning rate for model training.         |
| step_size     | number        | 7         | Step size for adjusting learning rate.     |
| gamma         | float      | 0.1       | Multiplicative factor for learning rate decay. |
| momentum      | float      | 0.9       | Momentum for Stochastic Gradient Descent (SGD) optimizer. |

Data Folder Requirements*

To effectively use the Easy Image Classifier, if you are using your own data, ensure your data is structured as follows:

```
   data/
   ├── train/
   │   ├── class_1/
   │   ├── class_2/
   │   ├── ...
   │   
   ├── val/
   │   ├── class_1/
   │   ├── class_2/
   │   ├── ...
```

Have approximately 80% of your data in the 'train' folder and the remaining 20% in the 'val' folder. This division is crucial for training and validation.

Inside the 'train' and 'val' subdirectories, create class-specific subfolders, and place images related to each class within them. The subfolder names should correspond to the class labels you wish to classify. For a visual representation of this data organization, you can also refer to the 'data' directory within the easy_image_classifier folder.