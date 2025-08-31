# app.py - Gradio app for Healthy Plate
import joblib
import pandas as pd
import gradio as gr
import numpy as np

# Load model and feature order
obj = joblib.load("healthy_plate_pipeline.joblib")
pipeline = obj["pipeline"]
FEATURES = obj["features"]

label_map = {1: "Balanced", 0: "Unbalanced"}

def get_suggestion(row):
    tips = []
    if row["Fat"] > 80:
        tips.append("Reduce fatty foods.")
    if row["Sugars"] > 50:
        tips.append("Lower sugar intake.")
    if row["Dietary Fiber"] < 20:
        tips.append("Add more fiber (fruits/vegetables).")
    if row["Protein"] < 50:
        tips.append("Increase protein intake.")
    if not tips:
        return "Your input looks generally balanced. Keep it up!"
    return " ".join(tips)

def predict(calories, protein, fat, carbs, sugars, fiber, sodium, nutrition_density):
    # Build input row with exact feature names/order
    row = {
        "Caloric Value": calories,
        "Protein": protein,
        "Fat": fat,
        "Carbohydrates": carbs,
        "Sugars": sugars,
        "Dietary Fiber": fiber,
        "Sodium": sodium,
        "Nutrition Density": nutrition_density
    }
    X = pd.DataFrame([row], columns=FEATURES)
    # predict
    try:
        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        confidence = round(max(proba) * 100, 2)
    except Exception as e:
        return f"Error in prediction: {e}"
    suggestion = get_suggestion(row)
    return f"Prediction: {label_map.get(pred,'?')} (Confidence: {confidence}%)\nAdvice: {suggestion}"

inputs = [
    gr.Number(label="Calories (kcal)", value=200),
    gr.Number(label="Protein (g)", value=20),
    gr.Number(label="Fat (g)", value=15),
    gr.Number(label="Carbohydrates (g)", value=40),
    gr.Number(label="Sugars (g)", value=10),
    gr.Number(label="Dietary Fiber (g)", value=5),
    gr.Number(label="Sodium (mg)", value=300),
    gr.Number(label="Nutrition Density (dataset units)", value=50),
]

description = """
Healthy Plate — a simple nutrition advisor. Enter approximate nutritional values (per serving) and get a quick Balanced/Unbalanced prediction, a confidence score, and short advice.
**Note:** This app is a prototype. Labels are heuristic and for demo purposes only.
"""

gr.Interface(fn=predict, inputs=inputs, outputs="text", title="Healthy Plate — Nutrition Advisor", description=description, examples=[
    [500, 30, 10, 50, 8, 6, 400, 60],
    [900, 10, 70, 100, 45, 5, 900, 30],
]).launch()
