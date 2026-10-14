# OptiCrop - Smart Agricultural Production Optimization Engine

OptiCrop is an advanced software system that utilizes data-driven insights to optimize agricultural production for different crops. By integrating key environmental factors such as Nitrogen (N), Phosphorous (P), Potassium (K) levels, soil temperature, humidity, pH, rainfall, and crop types, OptiCrop provides intelligent recommendations to farmers for maximizing yields and resource efficiency.

---

## 🌟 Key Features & Scenarios

### 🌾 Scenario 1: Smart Crop Recommendation for Farmers
Farmers enter soil nutrients (N, P, K) and localized climate parameters (temperature, humidity, pH, rainfall). OptiCrop calculates the optimal crop with confidence scores, top-3 alternative crop suggestions, soil health diagnostics, and print-ready agricultural advisory reports.

### 🔍 Scenario 2: Crop Suitability & Environmental Assessment
Farmers and agronomists evaluate whether specific land conditions are compatible with a target crop (from 22 supported varieties). The engine provides an environmental compatibility percentage, tolerance ratings (*Highly Suitable, Moderately Suitable, Marginally Suitable, Not Suitable*), and factor-by-factor gap advice.

### 📊 Scenario 3: Agricultural Research & Policy Planning Dashboard
A research dashboard designed for agricultural universities and policymakers. Features interactive feature importance rankings (*Rainfall 23.0%, Humidity 22.4%, Potassium 17.5%, etc.*), correlation heatmaps, agro-climatic regional zoning recommendations, and evidence-based policy guidelines.

### 🔐 User Management & Audit History
* Secure User Registration & Login with PBKDF2/SHA-256 password hashing.
* Historical prediction and suitability assessment logs.
* One-click **Export to CSV** for historical data analysis.
* **Dataset Explorer** to search and filter through the 2,200 curated agricultural samples.
* **User Feedback & Ratings** module.

---

## 🏗️ Technical Architecture & Tech Stack

* **Presentation Layer:** Responsive HTML5, Bootstrap 5, Font Awesome, Chart.js.
* **Application Layer:** Python 3.11, Flask Framework, Werkzeug security.
* **Machine Learning & Data Layer:** Scikit-learn (Random Forest Classifier: **99.55% accuracy**, **99.32% 5-fold CV**), Pandas, NumPy, SciPy.
* **Database:** SQLite relational database (`opticropt.db`).
* **Deployment:** WSGI / Gunicorn compatible (Render, Heroku, Railway, AWS, or Local).

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AQBAL-debug/OptiCrop-Smart-Agricultural-Optimization_deploy.git
cd OptiCrop-Smart-Agricultural-Optimization_deploy
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Training
Train and evaluate the Random Forest model:
```bash
python train.py
```

### 4. Run the Automated Test Suite (32 Test Cases)
```bash
python test_suite.py
```

### 5. Run the Application
Start the Flask web application:
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:5000`**

*(Streamlit backup app can also be run via `streamlit run app_streamlit.py`)*

---

## 🔑 Demo Login Accounts

| Role | Username / Email | Password |
| :--- | :--- | :--- |
| **Farmer** | `farmer` *(or `farmer@opticrop.com`)* | `password123` |
| **Admin** | `admin` *(or `admin@opticrop.com`)* | `admin123` |

---

## 🌐 Cloud Deployment Guide

### Deploy on Render (Recommended)
1. Fork or push this repository to GitHub.
2. Sign in to [Render](https://render.com) and click **New Web Service**.
3. Select your repository: `OptiCrop-Smart-Agricultural-Optimization_deploy`.
4. Configure:
   * **Environment:** Python 3
   * **Build Command:** `pip install -r requirements.txt && python train.py`
   * **Start Command:** `gunicorn app:app`
5. Click **Deploy Web Service**.

### Deploy on Railway / Heroku
* A valid [`Procfile`](Procfile) is included: `web: gunicorn app:app`.

---

## 👥 Academic Credits & Team

* **Curriculum:** APSCHE Technical Engineering Project 2025-2026
* **Team Lead:** Shaik Husamuddin
* **Members:** Mohammad Aqball, Nama Akhila, Peruboina Lakshmi Meghana, Madugula SudheerBabu
