import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns

#combining all dataset into 1 large dataset

columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
files = ["processed.cleveland.data",
"processed.hungarian.data","processed.switzerland.data", "processed.va.data"]



# Data collection
dfs = []
for file in files:
    temp_df = pd.read_csv(file, header=None, names=columns, na_values = "?")
    dfs.append(temp_df)
df = pd.concat(dfs, ignore_index = True)
print("combined dataset ", df.shape)
print(df.head())



# Data exploration
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.head())

# Checking Duplication
print("Duplicate Rows :",df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate Rows :",df.shape)
df = df.fillna(df.mean(numeric_only=True))

# converting into binary classification
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

print("\nBinary target distribution:")
print(df["target"].value_counts())

# Features and Targets
x = df.drop("target",axis=1)
y= df["target"]
print("\n feature shape", x.shape)
print("target_shape",y.shape)

#train and text split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5, random_state=32)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#EDA
sns.set_style("whitegrid")
features = ["age","trestbps","chol","thalach","oldpeak","ca"]
plt.figure(figsize=(15,10))
for i, feature in enumerate(features,1):
    plt.subplot(2,3,i)
    sns.boxplot(x="target", y=feature, data = df)
    plt.title(f"{feature} vs heart_disease")
    plt.xlabel("heart_disease_where_o=no,1=yes")
    plt.ylabel(feature)
plt.tight_layout()
plt.show()

#heat map
plt.figure(figsize=(10,8))
heat = df.corr(numeric_only=True)
sns.heatmap(heat, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.7)
plt.title("Heatmap")
plt.show()

# model selecting

#1st SVM


parameters_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100]
}

model_1 = GridSearchCV(
    SVC(probability=True, random_state=23),
    parameters_grid,
    scoring="accuracy",
    cv=6,
    n_jobs=-1
)

model_1.fit(x_train, y_train)

print("Best_Parameters:", model_1.best_params_)
print("Best_accuracy:", model_1.best_score_)

#2nd  KNN



parameters_grid = {
    "n_neighbors": [1, 5, 10, 15, 20],
    "metric": ["minkowski", "euclidean", "manhattan"]
}

model_2 = GridSearchCV(
    KNeighborsClassifier(),
    parameters_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

model_2.fit(x_train, y_train)

print("Best Parameters:", model_2.best_params_)
print("Best Accuracy:", model_2.best_score_)


if model_1.best_score_ > model_2.best_score_:
    best_model = model_1.best_estimator_
    print("SVM is the winner")
else:
    best_model= model_2.best_estimator_
    print("KNN is the winner")

# here is evaluation for best model which i indlude precision,recall,f1 score, confusion matrix

imputer = SimpleImputer(strategy="median")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



from sklearn import metrics
y_predicted= best_model.predict(x_test)

print("Test_accuracy :", accuracy_score(y_test,y_predicted))
print("precision  :", precision_score(y_test,y_predicted))
print("f1_score :", f1_score(y_test,y_predicted))
print("confusion_matrix :", confusion_matrix(y_test,y_predicted))
print("classification :", classification_report(y_test,y_predicted))

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring = "accuracy")

print("5-fold_cross_validation_scores :", cv_scores)
print("cv_accuracy :",cv_scores.mean())
print("standard_Deviation :", cv_scores.std())


import joblib
joblib.dump(best_model, "best_model")
joblib.dump(imputer, "imputer ")
joblib.dump(scaler, "scaler ")
print("Model_saved")



import streamlit as st

model = joblib.load("best_model")
imputer = joblib.load("imputer")
scaler = joblib.load("scaler")

st.set_page_config(page_title ="CHD_Prediction" )#page_icon=""
st.title("Coronary_Heart_Disease_prediction")
st.write("Enter_patient_details_here")

age = st.number_input("age", min_value = 5, max_value = 150)
sex = st.selectbox("sex",[0,1], format_func=lambda x:"Female" if x==0 else "Male")
cp = st.number_input("Chest_pain_type", min_value=0, max_value=5, value = 1)
chol = st.number_input("Cholesterol",min_value = 50, max_value = 700)# value = 100
trestbps = st.number_input("Blood_Pressure", min_value = 50, max_value =400 )#value = 120
fbs = st.selectbox("Blood_suger > 120" , [0,1])
thalach = st.number_input("Maximum_Heart_rate", min_value = 50, max_value = 300)#value = 150
exang = st.selectbox("Induced_anigma",[0,1])
oldpeak = st.number_input("oldpeak", min_value = 0.0, max_value = 15.0, step = 0.1)
ca = st.number_input("number_of_vessel", min_value = 0, max_value = 4)
thal = st.number_input("thal", min_value = 0, max_value = 4)#value = 2
restecg = st.number_input("ECG", min_value=0 , max_value=2, value=1)
slope = st.number_input("slope", min_value=0, max_value=3, value=1)



if st.button("Predict_CHD"):
    input_data = pd.DataFrame ([{
        "age":age,
        "sex":sex,
        "cp":cp,
        "trestbps":trestbps,
        "chol":chol,
        "fbs":fbs,
        "restecg":restecg,
        "thalach":thalach,
        "exang":exang,
        "oldpeak":oldpeak,
        "slope" : slope,
        "ca":ca,
        "thal":thal,
       
    }])
    
    input_data = input_data[list(imputer.feature_names_in_)]

    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    
    probability = model.predict_proba(input_data)[0][1]
    threshold = 0.2
    prediction = 1 if probability >= threshold else 0

    st.subheader("prediction_Result")

    if prediction ==1:
        st.error(f"CHD_DETECTE, PRABABILITY : {probability:.2f}")

    else:
        st.success(f"No_CHD_Detected, probability : {probability: 2f}")


# run with python -m streamlit run ALLCODE.py

