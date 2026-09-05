import pandas as pd
from sklearn.svm import SVC

emp = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\SVM\emp_prm.csv')
std = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\SVM\student.csv')

# Employees promotion prediction
xe = emp[['age','years_experience','performance_score','training_hours','projects_completed']]
ye = emp['promoted']

model = SVC()
model.fit(xe,ye)

prede = model.predict([[30,5,81,20,6]])

def emps():
    if(prede == 1):
        return 'The model predicts the employee will get a promotion'
    else:
        return 'The model predicts the employee will not get a promotion'

# Students passing data
xs = std[['study_hours','attendance','previous_score','sleep_hours','assignments_completed']]
ys = std['passed']

model_1 = SVC()
model_1.fit(xs,ys)

preds = model_1.predict([[2.7,73,57,6.7,6]])

def stds():
    if(preds == 1):
        return 'The model predicts the student will pass the exam'
    else:
        return 'The model predicts the student will not pass the exam'

# Print predictions

print(f'{emps()} \n\n{stds()}')



