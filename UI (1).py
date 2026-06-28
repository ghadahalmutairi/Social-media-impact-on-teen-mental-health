"""""

Ghadah Almutairi         - 441203736
Aknan Alshubrumi        - 441203459
Tala Albishri            - 441203732 
Elaf Aldubayan            - 441203675
Jawaher Alharbi          - 441203357
Rawnah Alhasson         - 441203480


"""""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

try:
    # We now load the logistic model
    lr_model = joblib.load("logistic_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    messagebox.showerror("Error", "Missing .pkl files. Run your training script first.")
    exit()

root = tk.Tk()
root.title("Depression Risk Screener (Logistic Regression)")
root.geometry("600x550")

canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas, padding="20")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def create_slider(parent, text, min_val, max_val, default):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=6)
    label = ttk.Label(frame, text=text, width=35, anchor="w", font=("Arial", 10))
    label.pack(side=tk.LEFT)
    slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, resolution=0.1 if isinstance(default, float) else 1)
    slider.set(default)
    slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)
    return slider

ttk.Label(scrollable_frame, text="Mental Health & Lifestyle Indicators", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 15))

stress_level = create_slider(scrollable_frame, "Stress Level (1-10)", 1, 10, 5)
anxiety_level = create_slider(scrollable_frame, "Anxiety Level (1-10)", 1, 10, 5)
addiction_level = create_slider(scrollable_frame, "Addiction Level (1-10)", 1, 10, 5)
sleep_hours = create_slider(scrollable_frame, "Avg. Sleep Hours", 3, 12, 7)
daily_social_media = create_slider(scrollable_frame, "Daily Social Media Hours", 0, 12, 3)
screen_time_bed = create_slider(scrollable_frame, "Screen Time Before Bed", 0, 6, 1)
physical_activity = create_slider(scrollable_frame, "Weekly Exercise (Hours)", 0, 14, 2)
academic_gpa = create_slider(scrollable_frame, "Academic Performance (GPA)", 0.0, 4.0, 3.0)

social_frame = ttk.Frame(scrollable_frame)
social_frame.pack(fill=tk.X, pady=6)
ttk.Label(social_frame, text="In-Person Social Interaction", width=35, anchor="w", font=("Arial", 10)).pack(side=tk.LEFT)
social_var = tk.StringVar(value="Medium")
social_menu = ttk.Combobox(social_frame, textvariable=social_var, values=["Low", "Medium", "High"], state="readonly")
social_menu.pack(side=tk.RIGHT, expand=True, fill=tk.X)

result_label = ttk.Label(scrollable_frame, text="", font=("Arial", 12, "bold"), anchor="center")
result_label.pack(fill=tk.X, pady=20)

def check_prediction():
    input_data = {
        'age': [16], 
        'daily_social_media_hours': [float(daily_social_media.get())],
        'sleep_hours': [float(sleep_hours.get())],
        'screen_time_before_sleep': [float(screen_time_bed.get())],
        'academic_performance': [float(academic_gpa.get())],
        'physical_activity': [float(physical_activity.get())],
        'stress_level': [int(stress_level.get())],
        'anxiety_level': [int(anxiety_level.get())],
        'addiction_level': [int(addiction_level.get())],
        'gender_male': [1], 
        'platform_usage_Instagram': [0],
        'platform_usage_TikTok': [0],
        'social_interaction_level_low': [1 if social_var.get() == "Low" else 0],
        'social_interaction_level_medium': [1 if social_var.get() == "Medium" else 0]
    }

    df_input = pd.DataFrame(input_data)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    
    # Scale input and use the Logistic Regression model
    df_input_scaled = scaler.transform(df_input)
    prediction = lr_model.predict(df_input_scaled)[0]

    if prediction == 1:
        result_label.config(text="Status: High Risk Detected", foreground="#CC0000")
    else:
        result_label.config(text="Status: No Significant Risk", foreground="#008000")

submit_btn = ttk.Button(scrollable_frame, text="Submit Assessment", command=check_prediction)
submit_btn.pack(fill=tk.X, pady=10)

root.mainloop()