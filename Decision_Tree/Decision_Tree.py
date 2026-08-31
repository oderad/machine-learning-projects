import pandas as pd
from sklearn.tree import DecisionTreeClassifier

tennis = pd.read_csv(r'C:\Users\tennis.csv')
scores = pd.read_csv(r'C:\Users\scores.csv')
ice_cream = pd.read_csv(r'C:\Users\ice_cream.csv')


# Tennis predictions
# Quantify  all the columns
tennis['Play_TMeasure'] = tennis['PlayTennis'].str.replace({'Yes':'1','No':'0'})
tennis['Play_TMeasure'] = pd.to_numeric(tennis['Play_TMeasure'], errors = 'coerce')
tennis['Outlook_Measure'] = tennis['Outlook'].str.replace({'Overcase':'0','Rainy':'1','Sunny':'2'})
tennis['Outlook_Measure'] = pd.to_numeric(tennis['Outlook_Measure'],errors='coerce')
tennis['Outlook_Measure'] = tennis['Outlook_Measure'].astype('Int64')
tennis['Temp_Measure'] = tennis['Temperature'].str.replace({'Hot':'1', 'Cool':'0'})
tennis['Temp_Measure'] = pd.to_numeric(tennis['Temp_Measure'], errors='coerce')

xt = tennis[['Outlook_Measure','Temp_Measure']]
yt = tennis['Play_TMeasure']

model = DecisionTreeClassifier()
model.fit(xt,yt)

predt = model.predict([[1,0]])

def ten():
    if(predt == 1):
        return 'play tennis'
    else:
        return 'not play tennis'

# Scores
scores['Pass_Measure'] = scores['Pass'].str.replace({'Yes':'1','No':'0'})
scores['Pass_Measure'] = pd.to_numeric(scores['Pass_Measure'],errors='coerce')

xs = scores[['StudyHours','Attendance']]
ys = scores['Pass_Measure']

model_1 = DecisionTreeClassifier()
model_1.fit(xs,ys)
preds = model_1.predict([[4,70]])

def scr():
    if(preds == 1):
        return 'the student may pass'
    else:
        return 'the student may not pass'
# Ice cream
ice_cream['buy_measure'] = ice_cream['BuyIceCream'].str.replace({'Yes':'1','No':'0'})
ice_cream['buy_measure'] = pd.to_numeric(ice_cream['buy_measure'],errors='coerce')

xi = ice_cream[['Temperature']]
yi = ice_cream['buy_measure']

model_2 = DecisionTreeClassifier()
model_2.fit(xi,yi)

predi = model_2.predict([[23]])

def ice():
    if(predi == 1):
        return 'may buy ice cream'
    else:
        return 'may not buy ice cream'

# Predictions
print(f'Our model predicts that when it is rainy & cool then people will {ten()}')
print(f'Our model predicts that students that study 4 hours, have 70% attendence, {scr()}')
print(f'When it is 23 degrees C our model predicts that people {ice()}')





