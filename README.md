# Chest X-Ray Pneumonia Detection using Deep Learning

# Project Overview
Pneumonia is a serious lung infection that requires early and accurate diagnosis to prevent severe complications. Chest X-ray imaging is one of the most widely used diagnostic tools for detecting Pneumonia, but manual interpretation is time-consuming and depends heavily on expert radiologists.  
This project presents a deep learning–based automated system to classify chest X-ray images as **Normal** or **Pneumonia**, helping to support faster and more reliable medical diagnosis.

---

# Dataset Used
- **Dataset Name:** Chest X-Ray Pneumonia Dataset  
- **Source:** Kaggle  
- **Data Type:** Medical image data (Chest X-ray images)  
- **Classes:**  
  - Normal  
  - Pneumonia  

The dataset is divided into training, validation, and testing folders.  
It is **highly imbalanced**, with significantly more Pneumonia images than Normal images, which closely represents real-world medical scenarios.

---

## Problem Statement
The objective of this project is to design and train a deep learning model that can:
- Accurately classify chest X-ray images
- Handle class imbalance effectively
- Achieve high recall for Pneumonia detection
- Generalize well on unseen medical data

---

## Data Preprocessing
The following preprocessing steps were applied:

1. **Image Rescaling**  
   Pixel values were normalized to the range [0, 1] to stabilize training.

2. **Data Augmentation**  
   To improve generalization and reduce overfitting:
   - Rotation
   - Zoom
   - Width and height shifts
   - Horizontal flipping  
   were applied to training images.

3. **Class Imbalance Handling**  
   Since the dataset is imbalanced, **class weights** were computed and applied during training to ensure the model does not become biased toward the majority class.

---

## Model Architecture
A **Convolutional Neural Network (CNN)** with **transfer learning** was used.

### Base Model
- **MobileNetV2** pretrained on ImageNet
- Used as a **feature extractor**
- Top layers removed (`include_top=False`)
- Base model weights frozen to preserve learned visual features

### Custom Classification Head
- Global Average Pooling
- Fully connected dense layer
- Dropout layer for regularization
- Sigmoid output layer for binary classification

This architecture provides a balance between **performance and efficiency**, making it suitable for medical image analysis.

---

## Training Strategy
- **Framework:** TensorFlow / Keras
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Hardware:** GPU (Google Colab)
- **Epochs:** 10–15
- **Class Weights:** Applied to handle imbalance

Training was performed with validation monitoring to ensure stable convergence and avoid overfitting.

---

## Model Evaluation
The trained model was evaluated on unseen test data using standard classification metrics:

- **Accuracy:** Overall correctness of predictions
- **Precision:** Reliability of Pneumonia predictions
- **Recall (Sensitivity):** Ability to detect actual Pneumonia cases
- **F1-Score:** Balance between precision and recall

A **confusion matrix** was generated to visualize:
- True Positives
- True Negatives
- False Positives
- False Negatives

Special emphasis was placed on **recall**, as missing a Pneumonia case is critical in medical diagnosis.

---

## Results
- **Test Accuracy:** ~87%
- **Recall:** High (effective Pneumonia detection)
- The model shows strong generalization and balanced performance despite dataset imbalance.

---

## Model Saving and Reusability
The trained model was saved in:
- **Native Keras format (`.keras`)** – recommended and future-proof
- **HDF5 format (`.h5`)** – legacy compatibility

The saved model was successfully reloaded and tested, demonstrating full reusability for inference and deployment.

---

## Sample Predictions
Visual predictions were generated on test images to verify real-world performance.  
The model correctly identified Normal and Pneumonia cases, confirming effective feature learning.

---

## Platform and Tools
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Development Environment:** Google Colab
- **Version Control:** GitHub
- **Hardware Acceleration:** GPU

---

## Limitations
- The model uses only X-ray images and does not consider patient clinical history.
- Performance may vary across different hospitals or imaging devices.
- Binary classification only (Normal vs Pneumonia).

---

## Future Scope
- Fine-tuning deeper layers of the pretrained model
- Multi-class lung disease classification
- Integration with clinical data
- Deployment as a web or mobile diagnostic tool

---

## Conclusion
This project demonstrates the effective use of deep learning and transfer learning for medical image classification. By combining CNNs with MobileNetV2 and addressing class imbalance, the model achieved strong performance on chest X-ray Pneumonia detection. The results highlight the potential of deep learning systems to assist healthcare professionals in accurate and timely diagnosis.

---

