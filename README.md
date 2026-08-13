# 🚦 SafeRoad AI

### Road Accident Severity Prediction using Machine Learning

SafeRoad AI is a machine learning-based web application that predicts the severity of a road accident based on environmental and traffic conditions.

The project uses a **Random Forest Classifier** trained on road accident data and provides an interactive interface built with **Streamlit**.

---

## 📌 Project Overview

Road accidents can be influenced by several factors such as weather conditions, road surface, lighting, road type, speed limits, number of vehicles, and number of casualties.

SafeRoad AI analyzes these factors and predicts the possible accident severity as:

- Slight
- Serious
- Fatal

The application allows users to enter accident-related conditions through an interactive web interface and receive an immediate prediction.

---

## 🎯 Objectives

- Explore and analyze road accident data.
- Perform data cleaning and preprocessing.
- Identify important patterns and relationships in the dataset.
- Apply machine learning for accident severity prediction.
- Build an interactive web application using Streamlit.
- Create a GitHub-ready data science project.

---

## 🧠 Machine Learning Model

The project uses:

**Random Forest Classifier**

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

### Model Performance

**Test Accuracy: approximately 85.17%**

> Note: The reported accuracy is based on the test split used during model development. It should not be interpreted as a guarantee of real-world prediction accuracy.

---

## 📊 Input Features

The application uses the following features:

| Feature | Description |
|---|---|
| Weather Conditions | Weather condition during the accident |
| Road Surface Conditions | Condition of the road surface |
| Light Conditions | Lighting condition at the time |
| Road Type | Type of road |
| Speed Limit | Speed limit of the road |
| Number of Vehicles | Number of vehicles involved |
| Number of Casualties | Number of casualties |

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Scikit-learn** – Machine learning
- **Joblib** – Model and encoder saving/loading
- **Streamlit** – Interactive web application
- **Jupyter Notebook** – Data exploration and analysis
- **VS Code** – Development environment

---

## 📂 Project Structure

```text
SafeRoad-AI/
│
├── app/
│   └── app.py
│
├── data/
│   └── Road Accident Data.csv
│
├── models/
│   ├── accident_model.pkl
│   └── encoders.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore