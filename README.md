# 📈 Stock Price Prediction using Machine Learning

This project predicts stock prices using **Linear Regression** and **LSTM (Long Short-Term Memory)** models.
A simple **Flask web application** is used to visualize predictions and stock trends.

---

## 🚀 Features

* 📊 Stock price prediction using Machine Learning
* 🤖 Two models used:

  * Linear Regression
  * LSTM Neural Network
* 📉 Interactive stock chart
* 🌐 Flask-based web interface
* 📁 Model metrics stored for evaluation

---

## 🛠️ Technologies Used

* Python
* Flask
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Chart.js
* HTML / CSS / JavaScript

---

## 📂 Project Structure

```
Stock-Prediction
│
├── app.py                # Flask web application
├── train_models.py       # Model training script
├── stock_lr_lstm.py      # Stock prediction logic
├── stock_data.csv        # Dataset
│
├── models
│   ├── lstm_model.keras
│   └── metrics.json
│
├── templates
│   └── index.html
│
├── static
│   └── chart.js
│
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository

```
git clone https://github.com/niteshsah3424/Stock-Prediction.git
```

Move into the project folder

```
cd Stock-Prediction
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
python app.py
```

Open in browser

```
http://127.0.0.1:5000
```

---

## 📊 Models Used

### Linear Regression

A basic regression model used to predict stock prices based on historical data.

### LSTM (Long Short-Term Memory)

A deep learning model designed for **time series prediction**, capable of learning long-term dependencies in stock price data.

---

## 📸 Application Preview

Stock price visualization and prediction results are displayed through an interactive web dashboard.

---

## 📌 Future Improvements

* Add real-time stock data using APIs
* Improve model accuracy
* Deploy the application online
* Add more deep learning models

---

## 👨‍💻 Author

**Nitesh Sah**

GitHub:
https://github.com/niteshsah3424

