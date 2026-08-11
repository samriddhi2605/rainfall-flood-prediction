# 🌧️ Rainfall-Based Flood Risk Prediction

A machine learning project that uses historical rainfall data to classify the risk of unusually high rainfall and potential flood conditions.

## 📌 Project Overview

This project analyzes historical monthly rainfall patterns and uses a Random Forest machine learning model to classify rainfall conditions into:

- 🟢 Normal Flood Risk
- 🔴 High Flood Risk

The model uses monthly rainfall values from January to December as input features.

## 🎯 Objective

The main objective is to develop a machine-learning-based system that analyzes rainfall patterns and provides an early indication of potential high flood-risk conditions.

## 📊 Dataset

**Source:** Government of India Open Government Data (OGD) Platform / India Meteorological Department (IMD)

**Dataset:** Sub-Divisional Monthly Rainfall from 1901 to 2017

The dataset contains historical monthly rainfall observations for Indian meteorological subdivisions.

For this project, the dataset was filtered for **Kerala**.

## ⚙️ Project Workflow

Historical Rainfall Data  
↓  
Data Cleaning  
↓  
Feature Engineering  
↓  
Flood-Risk Classification  
↓  
Random Forest Model  
↓  
Model Evaluation  
↓  
Rainfall-Based Prediction

## 🤖 Machine Learning Model

The project uses the **Random Forest Classifier**.

### Input Features

The model uses rainfall values for all 12 months:

- January
- February
- March
- April
- May
- June
- July
- August
- September
- October
- November
- December

### Output

**0 → Normal Flood Risk**

**1 → High Flood Risk**

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The Random Forest model achieved approximately **79% accuracy** on the test dataset.

## 🔍 Feature Importance

Feature-importance analysis was performed to identify which months contributed most to the model's classification.

The analysis showed that **July rainfall was one of the most influential features** in predicting high-risk conditions.

## 🧪 Example Prediction

Users can enter monthly rainfall values and receive a flood-risk prediction.

Example:

January: 50 mm  
February: 60 mm  
March: 70 mm  
April: 200 mm  
May: 250 mm  
June: 500 mm  
July: 1000 mm  
August: 800 mm  
September: 400 mm  
October: 200 mm  
November: 100 mm  
December: 50 mm

The model provides a predicted flood-risk classification and the probability of high-risk conditions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- GitHub

## 📁 Project Structure

rainfall-flood-prediction/

├── data/  
├── src/  
├── Flood_Prediction_Project.ipynb  
├── README.md  
└── requirements.txt

## 🚀 How to Run

1. Clone this repository.
2. Open `Flood_Prediction_Project.ipynb`.
3. Open the notebook in Google Colab or Jupyter Notebook.
4. Install the required libraries.
5. Run the cells sequentially.
6. Enter monthly rainfall values when prompted.
7. View the predicted flood-risk classification.

## ⚠️ Important Note

This project is an academic machine-learning demonstration and should not be considered an official flood-warning system.

Actual flood prediction requires additional factors such as river levels, soil moisture, drainage conditions, topography, weather forecasts and real-time rainfall data.

## 👩‍💻 Author

**Samriddhi Srivastava**

GitHub: **samriddhi2605**

---

⭐ If you find this project useful, consider giving the repository a star!
