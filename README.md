# Rainfall-Based Flood Risk Prediction

A machine-learning project that studies historical rainfall patterns in Kerala and classifies years with unusually high monsoon rainfall.

## Project idea

The project follows a simple pipeline:

Historical rainfall data → feature engineering → high-rainfall-risk classification → machine learning → evaluation → risk prediction.

This project is inspired by the general idea of rainfall-based flood classification, but the implementation is independently structured and uses a public rainfall dataset.

## Dataset

**Source:** Government of India Open Government Data (OGD) Platform / India Meteorological Department (IMD).

Dataset: **Sub-Divisional Monthly Rainfall from 1901 to 2017**.

The dataset contains monthly rainfall observations for Indian meteorological subdivisions. This project filters the dataset for Kerala.

The downloaded dataset is stored as:

```text
data/rainfall_data.csv
```

## Target definition

This is an educational rainfall-risk proxy, not an official flood label.

The model marks the historical years in the **top 25% of Kerala's JJAS (June–September) rainfall distribution** as:

- `0` → normal rainfall-risk class
- `1` → higher rainfall-risk class

Using a percentile rather than copying a fixed threshold makes the target dependent on the historical distribution of the selected dataset.

## Features

The model uses:

- May rainfall
- June + July rainfall
- July + August rainfall
- August + September rainfall
- May-to-June rainfall change
- Monsoon rainfall variability
- June's share of JJAS rainfall

## Models

Two models are compared:

1. Logistic Regression
2. Random Forest

The better-performing model on the chronological test period is selected.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- 5-fold cross-validation accuracy
- Confusion matrix

It also generates:

- `outputs/rainfall_trend.png`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png` when Random Forest is selected

## Installation

```bash
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/flood_prediction.py
```

## Project structure

```text
rainfall-flood-prediction/
│
├── data/
│   └── rainfall_data.csv
│
├── src/
│   └── flood_prediction.py
│
├── outputs/
│
├── README.md
└── requirements.txt
```

## Important limitation

The model does **not** predict real-time floods.

Monthly rainfall alone cannot capture all factors involved in actual flooding, including river discharge, soil saturation, drainage capacity, terrain, reservoir levels, localized cloudbursts, and short-duration rainfall intensity.

Therefore, the output should be interpreted as a **historical high-rainfall risk classification**, not an emergency warning system.

## Attribution

Dataset source:

Government of India Open Government Data (OGD) Platform / India Meteorological Department (IMD).

The project is an independent implementation created for educational and portfolio purposes. The dataset is used under its public-data availability, with attribution to the original government source.
