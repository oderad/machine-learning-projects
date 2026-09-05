import pandas as pd
from sklearn.neural_network import MLPClassifier

cust = pd.read_csv(r'C:\Users\cust.csv')
heart = pd.read_csv(r'C:\Users\heart.csv')

# Customer cluster classification
xc = cust[['age','annual_income','purchases_per_year','avg_purchase','website_visits','discount_usage']]
yc = cust['segment']

model = MLPClassifier()
model.fit(xc,yc)

predc = model.predict([[47,85000,48,100,125,0.13]])

def csc():
    if(predc == 2):
        return 'The model predicts that the customer has a platinum membership'
    elif(predc == 1):
        return 'The model predicts that the customer has a silver membership'
    else:
        return 'The model predicts that the customer has a basic membership'

# Heart disease predictions
xh = heart[['age','blood_pressure','cholesterol','heart_rate','exercise_hours','smoking']]
yh = heart['heart_disease']

model_1 = MLPClassifier()
model_1.fit(xh,yh)

predh = model_1.predict([[48,135,230,85,2,1]])

def hrt():
    if(predh == 1):
        return 'The model predicts that the patient is at risk of heart disease'
    else:
        return 'The model predicts that the patient is not at risk of heart disease'

# Predictions
print(csc())
print(hrt())
