# Capstone-Project
# Breast Cancer Prediction using Deep Learning
![Picture1](https://github.com/TonyN2060/Capstone-Project/assets/128211953/619cebf4-c8b2-46d1-bed8-4017ffef1b31)

## 1. Business Overview
Breast cancer ranks as the most common cancer globally and the second leading cause of cancer-related deaths. According to the World Health Organization (WHO), breast cancer is the primary cause of death among women aged 45-55 years with, affecting 1 in 8 women. Early detection and timely intervention play a pivotal role in the prognosis of breast cancer. When detected early, and if adequate diagnosis and treatment are available, the chances of survival increase significantly. Thus, the importance of early detection cannot be overstated, as it can lead to more effective treatments, reduce the need for aggressive intervention procedures, and substantially lower breast-cancer-related deaths.

### Objectives
1.  To develop a deep learning model using medical imaging data capable of efficient segmentation of breast masses in ultrasound images.

2. To identify critical parameters for breast cancer detection.

3. To implement a user-friendly interface for healthcare professionals to upload medical images and receive predictions.

4.  To develop a model with at least 90% specificity and 90% sensitivity for accurate predictions.

## 2.0 Data Understanding

The Breast Ultrasound Images Dataset comprises breast ultrasound images from 600 female patients aged between 25 and 75 years. The dataset consists of 780 images with an average image size of 500 × 500 pixels. The images are stored in PNG format.

![Distribution of Images in each class](https://github.com/TonyN2060/Capstone-Project/assets/128211953/f1bb1944-02a1-4668-a5fc-5d84667af02e)

The data is categorized into three sets:

1. Benign: This set contains 437 images.
2. Malignant: This set contains 210 images.
3. Normal: This set contains 133 images.


## 3.0 Data Preparation 

In this section, we will perform several preprocessing steps on the images to prepare them for training. This includes:
1. Addressing class imbalance
2. Resizing the images to a consistent size
3. Normalizing the pixel values to a range between 0 and 1
4. Creating labels for each class
5. Applying data augmentation techniques to increase the variability and size of the training dataset

## 4.0 Modeling 
1. Baseline CNN Model
2. Baseline CNN Model with layers
3. VGG16 Model

## 5.0 Model Evaluation
Based on the accuracy and recall scores, the baseline model is able to predict fairly well and outperforms the latter two with the highest accuracy and recall scores for the three classes. Despite the deeper architecture of the second model, it is not performing well across the classes. While the adapted VGG model performs better than the second model, its performance remains below that of the baseline model.

## Conclusion
1.   Our model's achieved sensitivity ABOVE 80% demonstrateS its potential to make a substantial positive impact on patient outcomes and healthcare decision-making in the realm of early breast cancer detection.
2. The training progression of the model indicates consistent improvement in its ability to classify breast cancer images. The steady decrease in training and validation loss also suggests model convergence and effective learning from the data.
3. Overall, our deep learning based breast cancer prediction system holds significant promise for improving detection and intervention in breast cancer cases.

## Reccomendations
1. Experiment with different model architectures and pretrained models like ResNet, Inception, or EfficientNet to enhance model performance.
2. Establish a feedack loop with clinicians where model predictions can be reviewed and corrected.
3. Expand the dataset with more diverse samples for increased model robustness.
4. Incorporate patient metadata and clinical parameters for improved accuracy.

## Deployment
The model was deployed using Streamlit and is working as expected.
