# Skin Lesion Analysis with Digital Image Processing

A university Digital Image Processing project that walks through a classic computer-vision pipeline on dermoscopic images from the **PH2 dataset**: mask generation, lesion segmentation, feature extraction and Random Forest classification.

> ## ⚠️ Not a diagnostic tool
> The notebook **creates a dummy `Target` column** (a random 50/50 split) when no real labels are present. The classifier is therefore trained on artificial labels, and the reported accuracy (about 0.65), sensitivity (0.6) and specificity (0.7) **carry no medical meaning**. The value of this project is the image-processing pipeline, not the classification result. Do not use it for any health decision.

## Pipeline

1. **Mount Google Drive** (the notebook is written for Google Colab).
2. **Explore the dataset**: walk the PH2 folders and load sample images.
3. **Generate masks**: create a lesion mask for each image and save it as `.png` in `generated_masks/`.
4. **Segment**: resize each image to 256 × 256, combine it with its mask using `cv2.bitwise_and`, and save the result in `Segmented_Images/`.
5. **Extract features**: convert each segmented image to grayscale, take the lesion pixels and compute statistical features. They are saved to `Feature_Extracted.xlsx`.
6. **Train**: `RandomForestClassifier` (200 trees), 80/20 stratified split.
7. **Evaluate**: confusion matrix, accuracy, sensitivity and specificity.

## Dataset

[PH2 dermoscopic image database](https://www.fc.up.pt/addi/ph2%20database.html): `.bmp` dermoscopic images with lesion masks and clinical annotations. The dataset is **not included** in this repository; download it yourself and respect its terms of use.

## Run it

1. Download PH2 and upload it to Google Drive.
2. Open `Projec_Skin_Cancer_Detection_2023_SE_01.ipynb` in Google Colab.
3. Edit `project_path` (and the dataset paths below it). They are hardcoded to `/content/drive/MyDrive/OEL/...` and will not exist on your Drive.
4. Run the cells in order: masks, segmentation, feature extraction, training.

Libraries: Python, OpenCV, NumPy, pandas, Matplotlib, scikit-learn.

## Limitations

- Labels are dummy labels, so the model has learned nothing clinical.
- Paths are tied to one Google Drive layout.
- No cross-validation, no deep-learning model, no clinical validation.

## Future improvements

- Join the real PH2 diagnosis labels (common nevus, atypical nevus, melanoma) to each image
- Add texture, shape and colour features
- Compare against a CNN
- Use cross-validation and report results per class

## Author

Hammad Ahmed, BS Software Engineering, [@Hammad-Ahmed-hk](https://github.com/Hammad-Ahmed-hk)
