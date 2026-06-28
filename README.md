# Social Media Impact on Mental Health

This repository contains the implementation of Machine Learning models to predict depression risk among teenagers based on behavioral, social, and lifestyle indicators.

## Team Members
* **Supervisor:** Dr. Renad Alsweed
* **Students:** 
  * Aknan Alshurumi
  * Rawnah Alhasson
  * Elaf Aldubayan
  * Ghadah Almutairi
  * Jawaher Alharbi
  * Tala Albishri

---

##  Project Objective
The main goal of this project is to build and evaluate Machine Learning models that predict depression risk among teenagers based on social, behavioral, and academic factors. More details can be found in the `project-presentation.pdf` file.

---

## Dataset & Preprocessing
* **Source:** Kaggle dataset containing 1,200 rows and 13 features.
* **Key Features:** Age, gender, stress/anxiety levels, academic performance, and daily social media usage.
* **Target Variable:** Depression / No Depression.

### Preprocessing Pipeline:
1. Cleaned missing values (nulls) and duplicate rows.
2. Extracted numerical values and filtered invalid records/age outliers.
3. Encoded categorical variables.
4. Normalized features using `StandardScaler`.

---

## 🤖 Models & Performance
The dataset was split using an **80/20 stratified split** to train and evaluate two models:

1. **Logistic Regression:**
   * Handled class imbalance using `class_weight='balanced'`.
   * Preferred for health screening due to its perfect recall.
   * Showed slightly more false positives compared to the decision tree.

2. **Decision Tree:**
   * Restricted to `max_depth=3` to avoid overfitting.
   * Highly interpretable and visually transparent.
   * Achieved higher overall precision with fewer classification errors.

---

## Graphical User Interface (GUI)
The project features a **Depression Risk Screener** application. Users can input personal lifestyle metrics (such as stress levels, sleep hours, and screen time) to receive an automated assessment risk status:
* **No Significant Risk**
* **High Risk Detected**

---

##  Future Enhancements
* Incorporating longitudinal studies to track shifts over time.
* Integrating data connectivity with wearable health devices.
* Deploying advanced Machine Learning architectures while maintaining strict data privacy.
