# 🫁 Tuberculosis Detection from Chest X-rays using Deep Learning & XAI

This project implements a **Tuberculosis (TB) detection system** using **Chest X-ray images**.
A **DenseNet121** deep learning model is trained to classify images as **Normal** or **Tuberculosis**.
To improve transparency and trust, **Explainable AI (XAI)** techniques such as **Grad-CAM** and **LIME** are integrated.

---

## 🚀 Features
- Binary classification: **Normal vs Tuberculosis**
- Deep learning model: **DenseNet121**
- Explainability:
  - 🔍 **Grad-CAM** for visual attention maps
  - 🧠 **LIME** for super-pixel based explanations
- Web application:
  - **FastAPI** backend
  - **React.js** frontend
- Can run **locally** or be **deployed online**

---

## 📊 Dataset
- **Name:** Tuberculosis (TB) Chest X-ray Dataset  
- **Source:** Kaggle  
- **Link:**  
  👉 https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset  

The dataset contains chest X-ray images classified into **Normal** and **Tuberculosis** categories and is used for training and evaluating the deep learning model.

---

## 🧠 Model Architecture
- **DenseNet121**
- Custom classifier head consisting of:
  - Fully connected (Dense) layer  
  - ReLU activation  
  - Dropout (to reduce overfitting)  
  - Output layer with **2 classes** (Normal / Tuberculosis)

DenseNet121 enables **feature reuse** by connecting each layer to every other layer in a feed-forward fashion, improving gradient flow and learning efficiency.

---

## 🔍 Explainable AI (XAI)

### ✅ Grad-CAM
- Uses gradients from the **last convolutional layer**
- Produces a **heatmap** highlighting lung regions that influenced the prediction
- Helps visualize **where** the model is “looking”

### ✅ LIME
- Divides the image into **superpixels**
- Randomly perturbs regions and observes prediction changes
- Produces **local explanations** that justify individual predictions

---

## ⚙️ Technologies Used

| Technology | Purpose |
|----------|--------|
| Python | Backend & ML development |
| PyTorch | Deep learning, training & inference |
| DenseNet121 | Feature extraction model |
| FastAPI | Backend REST API |
| React.js | Frontend user interface |
| OpenCV | Image processing |
| LIME | Model explainability |
| Grad-CAM | Visual interpretation |

---

## ▶️ How to Run Locally

### 1️⃣ Backend (FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app:app --reload --port 8000


### 1️⃣ Frontend (React)

cd frontend
npm install
npm start
