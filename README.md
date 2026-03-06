## LIVE APP LINK :-https://image-classifier-cifar10.streamlit.app/

# CIFAR-10 Image Classification using CNN (PyTorch)

A clean and simple Convolutional Neural Network built using **PyTorch** to classify images from the **CIFAR-10** dataset.  
This project includes model training, evaluation, dropout regularization, and visualization of training metrics.

---

## 📌 Project Overview

This project demonstrates how to build and train a CNN on the CIFAR-10 dataset.  
Key features include:

- Custom CNN architecture  
- Dropout regularization  
- Training & validation loops  
- Loss and accuracy tracking  
- GPU support (if available)

---

## 🧠 Model Architecture
```
Input: 3 × 32 × 32 (CIFAR-10 images)

[Conv2D → ReLU → MaxPool → Dropout]

[Conv2D → ReLU → MaxPool → Dropout]

[Conv2D → ReLU → MaxPool → Dropout]

Flatten

[Linear → ReLU → Dropout]

[Linear → Output (10 classes)]
```

---

## 📦 Requirements

Install dependencies:
```
pip install -r requirements.txt
```


**Key libraries used:**

- torch  
- torchvision  
- matplotlib  
- numpy  
- jupyter

---

## 🚀 How to Run

### 1. Clone the repository
```
git clone https://github.com/VarunSaxena07/pytorch-cifar10-model
```


### 2. Open the notebook

`jupyter lab`

Open `CNN.ipynb` and run all cells.

---

## 📊 Results

- Model trained for 10 epochs  
- Validation accuracy reached **~76%**  
- Smooth training curve with no overfitting  
- Dropout improved generalization  

You can visualize metrics using the included plotting code.

---

## 🗂 Project Structure

```
project/
│── CNN.ipynb
│── requirements.txt
│── README.md
└── .gitignore
```

---

## 🔥 Future Improvements

- Add Batch Normalization  
- Add Data Augmentation (RandomCrop, RandomFlip)  
- Switch to ResNet18 for 90%+ accuracy  
- Add Grad-CAM visualization  
- Deploy model with Flask / FastAPI  

---

## 🤝 Contributing

Feel free to open issues or pull requests for improvements!

---

## 📜 License

This project is released under the MIT License.
