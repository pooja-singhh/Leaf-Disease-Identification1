
# 🌿 Deep Learning-Based Leaf Disease Identification

## 📌 Project Overview
This project leverages **Deep Learning** to support agriculture by automatically detecting plant leaf diseases from images.  
The model is built using **EfficientNetB3** and trained on preprocessed agricultural datasets to achieve high accuracy in classification.  
The ultimate goal is to assist **farmers and agricultural experts** in early disease detection, enabling better crop management and reducing losses.

---

## ⚙️ Tech Stack
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Model Architecture**: EfficientNetB3 (Convolutional Neural Network)  
- **Programming Language**: Python  
- **Libraries**: NumPy, Pandas, Matplotlib, OpenCV (for preprocessing)  

---

## 📂 Dataset
- Agricultural leaf images dataset (preprocessed for classification tasks).  
- Includes multiple plant species with both **healthy** and **diseased** samples.  
- Images were resized, normalized, and augmented to improve model generalization.  


---

## 🔎 Methodology
1. **Data Preprocessing**
   - Image resizing & normalization  
   - Data augmentation (rotation, flipping, zooming, brightness adjustments)  

2. **Model Design**
   - Base Model: **EfficientNetB3** (transfer learning with pre-trained weights)  
   - Added fully connected layers with dropout for classification  

3. **Training**
   - Optimizer: Adam  
   - Loss Function: Categorical Cross-Entropy  
   - Metrics: Accuracy, Precision, Recall  

4. **Evaluation**
   - Train/Validation/Test split  
   - Confusion matrix & classification report for performance analysis  

---


## How to run this project?
1. Clone the repository:
```
git clone https://github.com/pooja-singhh/Leaf-Disease-Identification.git
cd leaf-disease-identification
```
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Train the model (optional, if not using pretrained weights):
```
python train.py
```

4.Run inference on a sample image:
```
 python predict.py --image path/to/leaf.jpg
```
## 🎯 Results
- Achieved **93% classification accuracy** on test data.  
- Model demonstrates strong potential for real-world application in agriculture.  

---


🏆 Impact

By leveraging AI in agriculture, this project provides a scalable and efficient solution for plant disease detection, helping farmers reduce losses and improve yield through early diagnosis and preventive action.
