"""""

Ghadah Almutairi         - 441203736
Aknan Alshubrumi        - 441203459
Tala Albishri            - 441203732 
Elaf Aldubayan            - 441203675
Jawaher Alharbi          - 441203357
Rawnah Alhasson         - 441203480


"""""


# Import Libraries
from matplotlib import cm
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
from imblearn.over_sampling import SMOTE 

# Load Dataset
df = pd.read_csv("C:/Users/Owner/OneDrive/المستندات/Teen_Mental_Health_Dataset2.csv")

print(df.shape)      # Number of rows and columns
print(df.columns)    # Column names
print(df.describe()) # Statistical summary
print(df.isnull().sum())  # Check if there is any null value

# Data cleaning
df.dropna(inplace=True)  # Remove missing values
print(df.isnull().sum())  # Check if any null values remain
print(df.dtypes)  # Check data types
df['stress_level'] = df['stress_level'].astype(int) 
df['anxiety_level'] = df['anxiety_level'].astype(int) 
print(df.dtypes)    
print(df.duplicated().sum())   
df_clean = df.drop_duplicates().copy()  # Remove duplicate rows 
print(df_clean.duplicated().sum())  

# Assign gender value
df_clean['gender'] = df_clean['gender'].str.strip().str.lower()
gender_mapping = {
    'm': 'male', 'male': 'male',
    'f': 'female', 'female': 'female'
}
df_clean['gender'] = df_clean['gender'].map(gender_mapping)

# Cleaning platform usage
def clean_platform(val):
    if not isinstance(val, str):
        return val
    val = val.strip().lower()
    if 'insta' in val:
        return 'Instagram'
    elif 'tik' in val:
        return 'TikTok'
    elif 'both' in val:
        return 'Both'
    return val

# Extract numeric values from social media hours column
df_clean['platform_usage'] = df_clean['platform_usage'].apply(clean_platform)
df_clean['daily_social_media_hours'] = df_clean['daily_social_media_hours'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
df_clean['daily_social_media_hours'] = pd.to_numeric(df_clean['daily_social_media_hours'], errors='coerce')

# Filter unrealistic values
df_clean = df_clean[df_clean['academic_performance'] <= 4.0]
df_clean = df_clean[(df_clean['age'] >= 13) & (df_clean['age'] <= 19)]

# Encode categorical variables
df_ml = pd.get_dummies(df_clean, columns=['gender', 'platform_usage', 'social_interaction_level'], drop_first=True)

# Separate features (X) and target (y)
X = df_ml.drop(columns=['depression_label'])
y = df_ml['depression_label']

# Split the data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale continuous features (Normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Initialize and Train the Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluate performance on Training and Test data
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display Training and Test Accuracy
print("\n-----")
print(f"Logistic Regression Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Logistic Regression Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification Report
print("\nLogistic Regression Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Depression', 'Depression']))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(
    cm,
    annot=True, 
    fmt='d', 
    cmap='Blues',
     xticklabels=['No Depression', 'Depression'],
    yticklabels=['No Depression', 'Depression'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Initialize and Train the Decision Tree model with reduced overfitting
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42)
dt_model.fit(X_train_res, y_train_res)

# Evaluate performance on Training and Test data
y_train_dt_pred = dt_model.predict(X_train_scaled)
y_test_dt_pred = dt_model.predict(X_test_scaled)

train_dt_accuracy = accuracy_score(y_train, y_train_dt_pred)
test_dt_accuracy = accuracy_score(y_test, y_test_dt_pred)

# Display Training and Test Accuracy for Decision Tree
print("\n-----")
print(f"Decision Tree Training Accuracy: {train_dt_accuracy * 100:.2f}%")
print(f"Decision Tree Test Accuracy: {test_dt_accuracy * 100:.2f}%")

# Decision Tree Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_test_dt_pred))

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, y_test_dt_pred)
sns.heatmap(
    cm_dt, 
    annot=True, 
    fmt='d', 
    cmap='Greens',
     xticklabels=['No Depression', 'Depression'],
    yticklabels=['No Depression', 'Depression']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()


# We use plt.subplots instead of plt.figure() to set figure size
fig, ax = plt.subplots(figsize=(10, 8))

# Create the tree plot
plot_tree(
    dt_model, 
    feature_names=X.columns.tolist(),        # Shows the feature names on each node
    class_names=['No Depression', 'Depression'],  # Labels the prediction categories
    filled=True,                             # Colors nodes based on the dominant class
    rounded=True,                            # Rounds node boxes for better visual appeal
    fontsize=6,                             # Keeps text readable
    ax=ax
)

# Add a title and save the figure
plt.title('Decision Tree Visual Representation', fontsize=16)
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the Logistic Regression model 
joblib.dump(model, "logistic_model.pkl") 
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
joblib.dump(scaler, "scaler.pkl")