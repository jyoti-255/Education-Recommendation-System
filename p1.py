#import lib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# Load the data
data = pd.read_csv("student-scores.csv")
print(data.head())

# Drop irrelevant columns
data.drop(columns=['id', 'first_name', 'last_name', 'email'], inplace=True)
print(data.head())

# Create new features from all scores
data["total_score"] = (data["math_score"] + data["history_score"] +
                       data["physics_score"] + data["chemistry_score"] +
                       data["biology_score"] + data["english_score"] +
                       data["geography_score"])
data["average_score"] = data["total_score"] / 7
print(data.head())

# Check unique career aspirations
print(len(data['career_aspiration'].unique()))
print(data.head())

# Encoding Categorical Columns
gender_map = {'male': 0, 'female': 1}
part_time_job_map = {False: 0, True: 1}
extracurricular_activities_map = {False: 0, True: 1}

data['gender'] = data['gender'].map(gender_map)
data['part_time_job'] = data['part_time_job'].map(part_time_job_map)
data['extracurricular_activities'] = data['extracurricular_activities'].map(extracurricular_activities_map)

career_aspiration_map = {
    'Lawyer': 0, 'Doctor': 1, 'Government Officer': 2, 'Artist': 3, 'Unknown': 4,
    'Software Engineer': 5, 'Teacher': 6, 'Business Owner': 7, 'Scientist': 8,
    'Banker': 9, 'Writer': 10, 'Accountant': 11, 'Designer': 12,
    'Construction Engineer': 13, 'Game Developer': 14, 'Stock Investor': 15,
    'Real Estate Developer': 16
}

data['career_aspiration'] = data['career_aspiration'].map(career_aspiration_map)
print(data.shape)

# Balance Dataset
print(data['career_aspiration'].value_counts())

# Drop rows with missing values in 'gender' and 'part_time_job'
data.dropna(subset=['gender', 'part_time_job'], inplace=True)

# Separate features and target variable
x = data.drop('career_aspiration', axis=1)
y = data['career_aspiration']

# Create SMOTE object
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

print(x_resampled.shape, y_resampled.shape)

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled)

# Models Training (Multiple Models)
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "Random Forest Classifier": RandomForestClassifier(),
    "K Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

for name, model in models.items():
    print("="*50)
    print("Model:", name)
    # Train the model
    model.fit(x_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(x_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)


#single input prediction
print("Predicted Label:",model.predict(x_test_scaled[10].reshape(1,-1))[0])
print("Actaul:",y_test.iloc[10])


import pickle
pickle.dump(scaler,open("scaler.pkl",'wb'))
pickle.dump(model,open("model.pkl",'wb'))

scaler=pickle.load(open("scaler.pkl",'rb'))
model=pickle.load(open("model.pkl",'rb'))

class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score,average_score):
    
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,total_score,average_score]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Predict using the model
    probabilities = model.predict_proba(scaled_features)
    
    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs


final_recommendations = Recommendations(gender='female',
                                        part_time_job=False,
                                        absence_days=2,
                                        extracurricular_activities=False,
                                        weekly_self_study_hours=7,
                                        math_score=65,
                                        history_score=60,
                                        physics_score=97,
                                        chemistry_score=94,
                                        biology_score=71,
                                        english_score=81,
                                        geography_score=66,
                                        total_score=534,
                                        average_score=76.285714)

print("Top recommended studies with probabilities:")
print("="*50)
for class_name, probability in final_recommendations:
    print(f"{class_name} with probability {probability}")












