import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import time

import warnings
warnings.filterwarnings('ignore')

# --- App Title and Description ---
st.title("Personal Fitness Tracker")
st.write("This WebApp predicts calories burned based on your fitness data.")

# --- Sidebar for User Input ---
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg): ", 25, 150, 60)
    height = st.sidebar.slider("Height (cm): ", 100, 210, 170)

    bmi_calculated = round(weight / ((height / 100) ** 2), 2)
    bmi = st.sidebar.slider("BMI: ", 15.0, 40.0, bmi_calculated)
    
    st.sidebar.text(f"Calculated BMI: {bmi_calculated:.2f}")
    st.sidebar.text("---")
    st.sidebar.text("Exercise Parameters: ")
    exercise_minutes = st.sidebar.slider("Minutes of Exercise: ", 0, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 200, 100)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 35, 42, 37)
    st.sidebar.text("---")
    
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender_male = 1 if gender_button == "Male" else 0
    gender_female = 1 if gender_button == "Female" else 0

    data_model = {
        "Gender_Female": gender_female,
        "Gender_Male": gender_male,
        "Age": age,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# --- Display User Parameters ---
st.header("Your Input Parameters")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# --- Data Loading and Preprocessing ---
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

age_groups = ["Young", "Middle-Aged", "Old"]
exercise_df["age_groups"] = pd.cut(exercise_df["Age"], bins=[20, 40, 60, 80], right=False, labels=age_groups)

bmi_category = ["Very severely underweight", "Severely underweight", "Underweight", "Normal", "Overweight", "Obese Class I", "Obese Class II", "Obese Class III"]
exercise_df["Categorized_BMI"] = pd.cut(exercise_df["BMI"], bins=[0, 15, 16, 18.5, 25, 30, 35, 40, 50], right=False, labels=bmi_category)
exercise_df["Categorized_BMI"] = exercise_df["Categorized_BMI"].astype("object")

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# --- Model Training with Hyperparameter Tuning ---
param_grid = {'n_estimators': [100, 500, 1000], 'max_features': [2, 3], 'max_depth': [4, 6, 8]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
random_reg = grid_search.best_estimator_

# --- Prediction ---
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# --- Prediction Display ---
st.header("Predicted Calories Burned")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(f"**{round(prediction[0], 2)} kilocalories**")

# --- Similar Results ---
st.header("Similar Results")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

# --- General Information ---
st.header("General Information")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"You are older than {round(sum(boolean_age) / len(boolean_age), 2) * 100}% of others.")
st.write(f"Your exercise duration is higher than {round(sum(boolean_duration) / len(boolean_duration), 2) * 100}% of others.")
st.write(f"Your heart rate is higher than {round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}% of others during exercise.")
st.write(f"Your body temperature is higher than {round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}% of others during exercise.")

# --- Recommendations ---
st.header("Personalized Recommendations")

predicted_calories = round(prediction[0], 2)

if predicted_calories < 200:
    st.write("🟢 Calorie burn is low. Increase workout intensity or duration.")
    st.write("🔹 Recommended: Brisk walking, light jogging, bodyweight exercises, yoga.")
elif 200 <= predicted_calories < 400:
    st.write("🟡 Calorie burn is moderate. Add strength training.")
    st.write("🔹 Recommended: Cycling, swimming, HIIT, strength training.")
elif 400 <= predicted_calories < 600:
    st.write("🟠 Calorie burn is good. Maintain consistency and push limits.")
    st.write("🔹 Recommended: Running, jump rope, rowing, circuit training, CrossFit.")
else:
    st.write("🔴 Calorie burn is high. Ensure proper nutrition and recovery.")
    st.write("🔹 Recommended: Endurance training, long-distance running, intense weightlifting, sports.")

st.write("🔥 **Tip:** Combine cardio and strength training.")

st.subheader("Exercise Recommendations based on BMI")
bmi_val = df["BMI"].values[0]
if bmi_val < 18.5:
    st.write("🔹 Underweight: Strength training, weight-gaining exercises, protein-rich diets.")
elif 18.5 <= bmi_val < 25:
    st.write("✅ Normal BMI: Balanced cardio, strength, flexibility (yoga).")
elif 25 <= bmi_val < 30:
    st.write("⚠️ Overweight: Running, swimming, HIIT, strength training.")
else:
    st.write("🚨 Obese: Low-impact exercises (walking, cycling, swimming).")

st.subheader("Recommendations based on Heart Rate")
heart_rate_val = df["Heart_Rate"].values[0]

if heart_rate_val < 60:
    st.write("🔸 Your heart rate is lower than normal. Consider increasing your workout intensity.")
elif 60 <= heart_rate_val <= 100:
    st.write("✅ Your heart rate is within a healthy range.")
elif 100 < heart_rate_val <= 120:
    st.write("🟡 Your heart rate is slightly elevated. Consider moderate intensity workouts.")
else:
    st.write("🔴 Your heart rate is high. Avoid high-intensity workouts and focus on light exercises.")

st.subheader("Recommendations based on Duration")
if df["Duration"].values[0] < 10:
    st.write("🔹 Increase workout duration to 30 minutes daily.")

st.subheader("Recommendations based on Age")
age_val = df["Age"].values[0]
if age_val < 40:
    st.write("🔹 Young: HIIT, sports activities.")
elif 40 <= age_val < 60:
    st.write("🔹 Middle-aged: Balanced cardio, strength, flexibility (yoga, Pilates).")
else:
    st.write("🔹 Older: Low-impact exercises (walking, swimming, chair exercises).")

st.write("🏋️ **Remember:** Consult a fitness expert before starting a new routine!")

# --- Additional Recommendations ---
st.header("Additional Tips")

st.write("💧 **Hydration:** Stay hydrated by drinking plenty of water before, during, and after workouts.")
st.write("🍎 **Nutrition:** Focus on a balanced diet with whole foods, lean protein, and plenty of fruits and vegetables.")
st.write("😴 **Rest:** Ensure adequate sleep for muscle recovery and overall well-being.")
st.write("🧘 **Stress Management:** Incorporate stress-reducing activities like meditation or deep breathing exercises.")
st.write("📈 **Progress Tracking:** Keep a journal or use a fitness tracker to monitor your progress and stay motivated.")