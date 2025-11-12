# CAPTCHA Recognition — README

This project implements a simple CAPTCHA recognizer for a fixed 5-character CAPTCHA format (A-Z, 0-9).

## Problem Summary
Each CAPTCHA:
- Has 5 characters (A-Z, 0-9)
- Fixed font, spacing, 
- Similar foreground & background colors and textures.
- No rotation or distortion

Provided 25 labeled CAPTCHA images for training. The task is to create a lightweight AI model to predict unseen CAPTCHA text.

## Approach

1. **Pre-process:** Isolate the individual characters from the 5-character image. Since the number of characters, font, and spacing are fixed, segmentation can be done reliably using fixed-width cropping. Standardize the segmented character images (e.g., resizing to a fixed dimension).

2. **Training:** Recognize the character in the segmented image. Given the small, fixed set of 36 classes (A-Z, 0-9), this is a simple multi-class classification problem. 

    Chosen Algorithm: Template Matching. Given the highly consistent nature of the captchas, a simple and  robust solution is to create a template (a normalized image) for each of the 36 possible characters from the training set and use template matching or a simple pixel-wise comparison (e.g., Euclidean distance) for recognition.
    
    An alternatiive solution is a simple Convolutional Neural Network (CNN). A small CNN (e.g., 2 convolutional layers, 1 dense layer) could be trained on the segmented characters for slightly better robustness against minor, unseen variations, although this requires more setup and dependencies (e.g., TensorFlow/PyTorch).

## Setup & Usage

1. **Install necessary libraries**

    Ensure OpenCV (cv2) and NumPy are installed:


        pip install opencv-python numpy
2 **Training**** 
    
    Run the training method using the provided sample captchas to generate character templates.
        solver = Captcha()
        solver.train(path_to_samples) # Run this once
3 **Inference**** 

    Initialize the Captcha class and call the instance method:

        solver(im_path='unseen_captcha.jpg', save_path='output.txt')
    
    The result is stored in the output.txt file.