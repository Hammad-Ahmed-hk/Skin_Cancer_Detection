**Skin Cancer Detection using Digital Image Processing**

This project focuses on detecting skin cancer using dermoscopic images from the PH2 dataset. The complete pipeline includes mask generation, lesion segmentation, feature extraction, and classification using a Random Forest model.

The work was developed as part of a Digital Image Processing course project.

Project Overview

Skin cancer detection can be improved with computer vision and machine learning. In this project:

Dermoscopic images are loaded from the dataset.

Lesion masks are generated.

Images are segmented using those masks.

Features are extracted from segmented lesions.

A Random Forest classifier is trained to classify lesions.

The notebook runs step by step from preprocessing to evaluation.

Dataset

Dataset used: PH2 Skin Lesion Dataset

Image format: .bmp

Each case contains:

Dermoscopic image

Associated lesion mask

The dataset was accessed from Google Drive inside Google Colab.

Project Pipeline
1. Mount Google Drive

The notebook mounts Google Drive to access:

Dataset images

Generated masks

Extracted features

Saved outputs

2. Dataset Exploration

The code:

Navigates dataset directories

Lists IMD folders

Identifies dermoscopic image folders

Loads sample images

This ensures correct path handling before processing.

3. Mask Generation

Masks are generated and saved in a separate folder:

generated_masks/


For each image:

Mask is created

Saved as .png

Used later for segmentation

4. Image Segmentation

Each dermoscopic image is:

Resized to 256 × 256

Combined with its corresponding mask

Segmented using cv2.bitwise_and

Segmented images are saved in:

Segmented_Images/


This step isolates the lesion from the background.

5. Feature Extraction

For each segmented image:

Converted to grayscale

Lesion pixels extracted

Statistical features calculated

Features are saved into an Excel file:

Feature_Extracted.xlsx


These features are later used for model training.

6. Model Training

The classification model used:

Random Forest Classifier

200 estimators

80/20 train-test split

Stratified sampling

Libraries used:

scikit-learn

pandas

numpy

Important Note About Labels

The notebook creates a dummy Target column if one does not exist.

This means:

The model is trained on artificial labels.

The current accuracy does not represent real medical performance.

For real-world use:

Proper ground truth labels must be added.

Diagnosis information should be linked to each image.

7. Evaluation Metrics

The following metrics are calculated:

Confusion Matrix

Accuracy

Sensitivity (Recall for malignant class)

Specificity

These help evaluate classification performance.

Technologies Used

Python

OpenCV

NumPy

Pandas

Matplotlib

Scikit-learn

Google Colab

Folder Structure
Project/
│
├── PH2Dataset/
│   ├── PH2 Dataset images/
│   ├── generated_masks/
│
├── Segmented_Images/
│
├── Feature_Extracted.xlsx
│
└── Skin_Cancer_Detection.ipynb

How to Run

Upload dataset to Google Drive.

Open the notebook in Google Colab.

Update the project_path if needed.

Run cells step by step:

Mask generation

Segmentation

Feature extraction

Model training

Limitations

Uses manually created or dummy labels.

No deep learning model.

No cross-validation.

No real clinical validation.

Future Improvements

Use real diagnosis labels.

Add more advanced features (texture, shape, color histogram).

Apply CNN-based deep learning models.

Deploy as a web application.

Add performance comparison between models.

Author

Hammad Ahmed
BS Software Engineering
Digital Image Processing Project
