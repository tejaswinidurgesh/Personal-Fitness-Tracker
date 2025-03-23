# Personal-Fitness-Tracker
A web application that to know the report of the personal fitness individually.
# Personal Fitness Tracker

## Overview
The **Personal Fitness Tracker** is a machine learning-based web application that estimates the number of calories burned based on user-provided fitness parameters. It uses a **Random Forest Regressor** optimized with **GridSearchCV** to enhance prediction accuracy. The application is built with **Streamlit** for an interactive and user-friendly experience.

## Features
- User inputs fitness parameters such as **age, weight, height, BMI, exercise duration, heart rate, and body temperature**.
- **Real-time calorie burn estimation** using a trained machine learning model.
- **Personalized fitness recommendations** based on calorie expenditure.
- **Comparison with similar users** for better insights.
- **Web-based interface** using Streamlit for easy accessibility.

## Technology Stack

### Programming Language
- Python

### Machine Learning Libraries
- **Scikit-learn**: Model training, hyperparameter tuning, and evaluation.
- **NumPy & Pandas**: Data preprocessing and manipulation.
- **Matplotlib & Seaborn**: Data visualization and exploratory analysis.

### Web Framework
- **Streamlit**: Interactive web-based UI for user input and result display.

### Data Storage
- **CSV files**: Used for storing fitness and calorie data.
- **SQLite / MySQL (Optional)**: Can be used for structured database storage in future updates.

### Deployment & Version Control
- **GitHub**: Version control and code collaboration.
- **Heroku / AWS / Google Cloud**: Future deployment for cloud-based accessibility.

## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/personal-fitness-tracker.git
cd personal-fitness-tracker

###On Windows
python -m venv venv
venv\Scripts\activate

### python3 -m venv venv
source venv/bin/activate

### To come out of the activation of virtual environment 
deactivate

### install required libraries using this text file after creating it
pip install -r requirements.txt

## To run the project after intalling required libraries
streamlit run app.py

### License
This project is licensed under the MIT License.
