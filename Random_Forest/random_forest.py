import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

scores = pd.read_csv(r'C:\Users\scores.csv')
loan = pd.read_csv(r'C:\Users\loan.csv')
fruit = pd.read_csv(r'C:\Users\fruit.csv')

# Student scores
scores['Pass_Measure'] = scores['Pass'].str.replace({'Yes':'1','No':'0'})
scores['Pass_Measure'] = pd.to_numeric(scores['Pass_Measure'], errors= 'coerce')

xs = scores[['StudyHours','Attendance','PreviousScore']]
ys = scores['Pass_Measure']

model = RandomForestClassifier()
model.fit(xs,ys)

preds = model.predict([[4,75,70]])

def scr():
    if(preds == 1):
        return 'the student passes'
    else:
        return 'the student does not pass'
# Loan approval
loan['Approved_Measure'] = loan['Approved'].str.replace({'Yes':'1','No':'0'})
loan['Approved_Measure'] = pd.to_numeric(loan['Approved_Measure'], errors='coerce')

xl = loan[['Age','Income','CreditScore','LoanAmount']]
yl = loan['Approved_Measure']

model_1 = RandomForestClassifier()
model_1.fit(xl,yl)

predl = model_1.predict([[40,40000,600,12000]])

def ln():
    if(predl == 1):
        return 'will be approved'
    else:
        return 'will not be approved'
# Fruit prediction
fruit['Type_Measure'] = fruit['Type'].str.replace({'Apple':'0','Banana':'1','Orange':'2'})
fruit['Type_Measure'] = pd.to_numeric(fruit['Type_Measure'],errors='coerce')

xf = fruit[['Weight','ColorCode','Sweetness','Size']]
yf = fruit['Type_Measure']

model_2 = RandomForestClassifier()
model_2.fit(xf,yf)

predf = model_2.predict([[200,4,7,7]])

def frt():
    if(predf == 0):
        return 'fruit is an apple'
    elif(predf == 1):
        return 'fruit is a banana'
    else:
        return 'fruit is an orange'

# Predictions
print(f'The model predicts that {scr()}')
print(f'The model predicts that the loan {ln()}')
print(f'The model predicts the {frt()}')
