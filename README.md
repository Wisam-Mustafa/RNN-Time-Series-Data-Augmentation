# Neural Networks with Sliding Window Data Augmentation for Time Series Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## ⚠️ Intellectual Property Notice
> **Note:** This repository serves as technical documentation for my Master's Thesis conducted at **Ural Federal University**. Due to Intellectual Property (IP) regulations regarding the dataset and source code, the raw code cannot be shared publicly. This documentation outlines the methodology, model architecture, and achieved results.

## 📄 Abstract
This project explores the impact of **Sliding Window Data Augmentation** on the performance of various Recurrent Neural Network (RNN) architectures for Human Activity Recognition (HAR). The study evaluates models based on **LSTM**, **SimpleRNN**, **GRU**, and a novel **Hybrid RNN**, applied to the classification of five outdoor activities: *Biking, Roller Skiing, Running, Skiing, and Walking*.

## 🚀 Key Features
*   **Data Augmentation:** Implemented a sliding window technique to address data scarcity and class imbalance.
*   **Hybrid Architecture:** Designed a model combining SimpleRNN, GRU, and LSTM layers.
*   **High Performance:** Achieved **93% Overall Accuracy** and **0.94 Macro F1-Score**.

## 🛠️ Methodology & Workflow
The system processes multivariate time-series data (Heart Rate, Speed, Altitude) through specific preprocessing pipelines, segmentation, and finally into the Neural Network.

### System Overview
![System Overview](Figure_5.png) 
*(Figure 5: Illustrative overview of loading, pre-processing, and window segmentation)*

## 📊 Data Analysis
The dataset comprises multivariate time series of outdoor sport activities. Below is the exploratory analysis showing temporal patterns.

![Data Analysis](Figure_3.png)

## 🧠 Model Architecture (Hybrid RNN)
The best-performing model utilized a hybrid approach:
1.  **Input Layer:** (DIMS, SLEN)
2.  **Parallel Branches:** SimpleRNN, GRU, and LSTM layers running in parallel.
3.  **Concatenation:** Merging features from all recurrent layers.
4.  **Dense Layers:** Fully connected layers with Dropout and Batch Normalization.
5.  **Output:** Softmax classification for 5 activities.

## 📈 Results
The **Hybrid RNN** model with a window size of **256** and step size of **128** yielded the best results, outperforming baseline models significantly.

### Performance Metrics
![Results Table](Table_15.png)
*(Table 15: Performance Metrics of Hybrid RNN Model)*

### Confusion Matrix
The model showed exceptional accuracy in distinguishing between complex activities like "Running" and "Walking".

![Confusion Matrix](Figure_9.png)

## 💻 Tech Stack
*   **Language:** Python
*   **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
*   **Tools:** Jupyter Notebook.

