# House Price Prediction & Model Comparison

## 📌 Project Overview

This project was developed as an **individual machine learning assignment**. The goal is to select a real-world problem, compare the performance of two machine learning algorithms using a public dataset, and deploy the best-performing model with a simple user interface that responds to user queries.

### 🎯 Problem Statement

Predict **house prices** based on property features such as area, number of bedrooms, bathrooms, parking spaces, and amenities (e.g., air conditioning, main road access).

### 📊 Dataset

* **Source:** Kaggle (Housing Price Dataset)
* **File:** `Housing.csv`
* **Type:** Supervised regression dataset
* **Target variable:** `price`

The dataset contains both numerical and categorical features related to residential properties.

---

## 🧠 Machine Learning Approach

### Algorithms Used

Two regression algorithms were selected and compared:

1. **Linear Regression**

   * Simple and interpretable baseline model
   * Assumes a linear relationship between features and house price

2. **Random Forest Regressor**

   * Ensemble-based model
   * Captures non-linear relationships
   * More robust to feature interactions

### Preprocessing Steps

* Converted categorical binary features (Yes/No) into numerical values (1/0)
* Applied **Standard Scaling** to numerical features
* Split dataset into training and testing sets

### Evaluation Metrics

The models were evaluated using:

* **R-Squared (R²)** – measures how well the model explains variance
* **Mean Absolute Error (MAE)** – measures average prediction error

### Model Selection

After comparison, the **best-performing model** was selected based on:

* Higher R² score
* Lower MAE

The selected model was saved using `joblib` and used in the user interface.

---

## 🖥️ User Interface

A **Streamlit-based web interface** was created to allow users to:

* Input property details
* Receive an instant house price prediction
* View model comparison results

The UI makes the ML model interactive and easy to use for non-technical users.

---

## 📂 Project Structure

```
├── Housing.csv           # Dataset
├── model_training.py     # Data preprocessing, training, evaluation, model saving
├── app.py                # Streamlit user interface
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Prerequisites

Make sure you have the following installed:

* **Python 3.9 or higher**
* **pip** (Python package manager)

You can check using:

```bash
python --version
pip --version
```

---

### 2️⃣ Clone or Download the Project

```bash
git clone <repository-url>
cd house-price-prediction
```

Or download the ZIP and extract it.

---

### 3️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows:**

```bash
venv\Scripts\activate
```

* **macOS / Linux:**

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

Install required libraries:

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

Alternatively, if `requirements.txt` is provided

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project Locally

### Step 1: Train the Models

Run the training script to preprocess data, train models, evaluate performance, and save the best model:

```bash
python model_training.py
```

This will:

* Load the dataset
* Train Linear Regression and Random Forest models
* Compare performance
* Save the selected model

---

### Step 2: Launch the User Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open automatically in your browser (usually at `http://localhost:8501`).

---

## 🧪 Example User Interaction

* User enters house details (area, bedrooms, amenities)
* Clicks **Predict Price**
* App displays the estimated house price using the trained ML model

---

## ✅ Assignment Requirements Mapping

| Assignment Requirement      | Implementation                     |
| --------------------------- | ---------------------------------- |
| Select a real-world problem | House price prediction             |
| Use Kaggle dataset          | Housing.csv                        |
| Compare two ML algorithms   | Linear Regression vs Random Forest |
| Evaluate performance        | R² and MAE                         |
| Select best model           | Based on evaluation metrics        |
| Create user interface       | Streamlit web app                  |

---

## 🚀 Conclusion

This project demonstrates the **end-to-end machine learning workflow**, from data selection and preprocessing to model evaluation and deployment. It highlights how ML models can be compared and integrated into a simple, interactive application.

---
