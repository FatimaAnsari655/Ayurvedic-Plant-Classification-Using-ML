# Ayurvedic Plant Classification System Using ML

## 🌿 Overview
**The Digital Vaidya** is a machine learning-based system designed to identify medicinal plants by analyzing images of their leaves. This project leverages deep learning to bridge traditional Ayurvedic knowledge with modern computer vision technology, specifically catering to the identification of 45 different plant species.

## 🚀 Key Features
*   **Massive Dataset**: Built and trained on a dataset comprising 17,500 images.
*   **Deep Learning Model**: Utilizes a Convolutional Neural Network (CNN) implemented via Keras and TensorFlow.
*   **High Accuracy**: The system includes a finalized model (`plant_model_final.keras`) optimized for classification tasks.
*   **Automated Preprocessing**: Includes robust scripts for image augmentation and dataset splitting.

## 📂 Project Structure
*   `Data_Preprocessing.ipynb`: Notebook for data cleaning, visualization, and preparation.
*   `Model_Training.ipynb`: Documentation of the model architecture, training loops, and evaluation metrics.
*   `app1.py`: The main Python application file for running the classification system.
*   `requirements.txt`: List of all necessary Python libraries to reproduce the environment.
*   `plant_model_final.keras`: The saved, high-accuracy trained model ready for deployment.

## 🛠️ Technologies Used
*   **Programming Language**: Python
*   **Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV, PIL
*   **Tools**: Jupyter Notebook, Anaconda, Git
*   MobileNetV2 (Transfer Learning)
*   Streamlit (Frontend)

  ## ⚙️ Methodology
1. Data preprocessing (resizing, normalization)
2. Data augmentation (rotation, zoom, flip)
3. Model building using MobileNetV2
4. Training and validation of the model
5. Prediction of plant name
6. Display of medicinal uses

## 🤖 Model Details
- Pre-trained model: MobileNetV2
- Input size: 224 × 224
- Activation: ReLU, Softmax
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

## 🖥️ Application
- Built using Streamlit
- User uploads an image of a plant leaf
- Model predicts plant name
- Displays medicinal uses of the plant
  
## 📂 Dataset
- Source: Kaggle
- Contains images of **45 different Ayurvedic plants**
- Leaf images are used for classification
- Images are resized to **224 × 224**
- Dataset is split into training, testing and validation sets

---
## 📖 How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/FatimaAnsari655/Ayurvedic-Plant-Classification-Using-ML.git

Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
python app1.py   

## Results
- The model successfully classifies Ayurvedic plants from leaf images
- Provides accurate predictions with medicinal information

## Author
- Fatima Ansari

