import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

student = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Nearest_Neighbors\student.csv')
height = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Nearest_Neighbors\height.csv')
exam = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Nearest_Neighbors\exam.csv')

# Students
student['Result_Measure'] = student['Result'].str.replace({'Pass':'1','Fail':'0'})
student['Result_Measure'] = pd.to_numeric(student['Result_Measure'], errors='coerce')

xs = student[['Hours_Studied','Attendance']]
ys = student['Result_Measure']

model = KNeighborsClassifier(n_neighbors= 3)
model.fit(xs,ys)

preds = model.predict([[8,90]])

def std():
    if(preds==1):
        return 'the student will pass'
    else:
        return 'the student will not pass'
# Height
height['Category_Measure'] = height['Category'].str.replace({'Small':'0','Medium':'1','Large':'2'})
height['Category_Measure'] = pd.to_numeric(height['Category_Measure'], errors='coerce')

xh = height[['Height','Weight']]
yh = height['Category_Measure']

model_1 = KNeighborsClassifier(n_neighbors=3)
model_1.fit(xh,yh)

predh = model_1.predict([[158,53]])

def ht():
    if(predh == 0):
        return 'is small'
    elif(predh == 1):
        return 'is medium'
    else:
        return 'is large'
# Exam performance
exam['Result_Measure'] = exam['Result'].str.replace({'Pass':'1','Fail':'0'})
exam['Result_Measure'] = pd.to_numeric(exam['Result_Measure'], errors='coerce')

xe = exam[['Study_Hours','Sleep_Hours','Previous_Score']]
ye = exam['Result_Measure']

model_2 = KNeighborsClassifier(n_neighbors=3)
model_2.fit(xe,ye)

prede = model_2.predict([[3,7,58]])

def exm():
    if(prede == 1):
        return 'the student will pass'
    else:
        return 'the student will fail'
    

# Predictions
print(f'The model predicts that {std()}')
print(f'The model predicts that the person {ht()}')
print(f'The model predicts that {exm()}')