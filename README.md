# 🏠 Bengaluru House Price Predictor

A full-stack Machine Learning application that predicts house prices in Bengaluru based on user inputs such as location, square footage, BHK, and bathrooms.

---

## 🌐 Live Demo

* 🔗 Frontend (Streamlit App): *(Add your Streamlit URL here)*
* 🔗 Backend API: https://bengaluru-house-price-predictor-gt52.onrender.com
* 📄 API Docs: https://bengaluru-house-price-predictor-gt52.onrender.com/docs

---

## 🚀 Features

* 💰 Real-time house price prediction
* 📍 Dynamic location selection via API
* 📊 Interactive dashboard with charts
* 🔗 FastAPI backend + Streamlit frontend integration
* 🌍 Deployed using Render and Streamlit Cloud

---

## 🖥️ Application UI

The application is divided into four main sections:

### 🏠 Home

* Overview of the application
* Dataset description
* Features and usage guidance

### 💰 Predict Price

* User inputs:

  * Location
  * Total Square Feet
  * BHK
  * Bathrooms
* Sends data to FastAPI endpoint `/predict_price`
* Displays predicted house price

### 📊 Dashboard

* Visual insights from the dataset:

  * Average price by location
  * Price vs BHK analysis
  * Price distribution histogram
  * Sqft vs price trends

### 🗂️ Sample Data

* Fetches sample dataset from API `/sample_dataframe`
* Displays structured data in tabular format

---

## 🧠 Tech Stack

### 🔹 Backend

* FastAPI
* Uvicorn
* Scikit-learn

### 🔹 Frontend

* Streamlit
* Plotly

### 🔹 Data Processing

* Pandas
* NumPy

### 🔹 Deployment

* Render (Backend API)
* Streamlit Cloud (Frontend)

---

## 📂 Project Structure

```id="k4r9tw"
project/
│── backend/
│     ├── app.py
│     ├── RF_model_pipeline.pkl
│     ├── cleaned.csv
│
│── frontend/
│     ├── app.py
│
│── requirements.txt
│── README.md
```

---

## ⚙️ Project Setup

### 1. Create Virtual Environment

```id="q7m2ax"
conda create -p venv python==3.12 -y
```

### 2. Activate Environment

```id="v9z1kd"
conda activate venv
```

### 3. Install Dependencies

```id="x5b8nr"
pip install -r requirements.txt
```

---

## ▶️ Run Locally

### Start FastAPI Backend

```id="u3n6hp"
uvicorn backend.app:app --reload
```

### Run Streamlit Frontend

```id="r2c8yf"
streamlit run frontend/app.py
```

---

## 🔌 API Endpoints

| Endpoint             | Method | Description                 |
| -------------------- | ------ | --------------------------- |
| `/predict_price`     | POST   | Predict house price         |
| `/cleaned_dataframe` | GET    | Get full cleaned dataset    |
| `/unique_locations`  | GET    | Get all available locations |
| `/sample_dataframe`  | GET    | Get sample dataset          |

---

## 🧪 Example Request

```id="b6k3xt"
POST /predict_price
```

```json id="n8p4sz"
{
  "location": "Whitefield",
  "total_sqft": 1200,
  "bhk": 2,
  "bath": 2
}
```

---

## ⚠️ Notes

* Render free tier may take time to wake up (cold start)
* Ensure correct API URL is used in frontend
* Use proper file paths for deployment

---

## 🚀 Future Improvements

* Add authentication
* Improve model accuracy
* Add advanced analytics dashboard
* Optimize API response time

---

## 👨‍💻 Author

**Taj Shaik**
Python Developer | Data Analyst | GenAI Enthusiast

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
