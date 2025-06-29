import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#Loading the dataset
df=pd.read_csv("placedata.csv")

#Analyzing the dataset
df.head()
df.describe()
df.shape
df.info()
df.isna().sum()

# Encode categorical variables
le=LabelEncoder()
cols=["ExtracurricularActivities","PlacementTraining","PlacementStatus"]
for i in cols:
    df[i]=le.fit_transform(df[i])
df.head().T
df.corr()

# Features and target
x=df[["AptitudeTestScore","HSC_Marks"]]
y=df[["PlacementStatus"]]

# Split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)

#LINEAR REGRESSION
lr=LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
lr.intercept_
ypred=lr.predict(x_test)
ypred
r2_score(y_test,ypred)

# LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(x_train, y_train)
ylr_pred = model.predict(x_test)
print("✅ Accuracy:", accuracy_score(y_test, ylr_pred))

#RandomForestClassifier
classify= RandomForestClassifier(n_estimators= 100, criterion="entropy")
classify.fit(x_train, y_train)
ypred=classify.predict(x_test)
a=accuracy_score(ypred,y_test)
print(a)

#KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3,metric='euclidean')
knn.fit(x_train,y_train)
ypred=knn.predict(x_test)
ypred
print(accuracy_score(y_test,ypred))

# Visualization
accuracies = {
    'Linear Regression (R²)': 0.36,
    'Logistic Regression': 0.78, 
    'Random Forest': 0.76,
    'K-Nearest Neighbors': 0.73
}
models = list(accuracies.keys())
scores = list(accuracies.values())
plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=['skyblue', 'lightgreen', 'salmon', 'plum'])
plt.ylim(0, 1.1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy / R² Score')
plt.xlabel('Algorithms')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
