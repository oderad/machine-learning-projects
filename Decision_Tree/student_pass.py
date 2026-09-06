import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Users\student_pass.csv')

x = data[['study_hours','attendance_percent','previous_score','sleep_hours','assignments_completed']]
y = data['passed']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.25
)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

predict = model.predict(x_test)
accuracy = accuracy_score(y_test,predict)

print(accuracy)

def prd():
    a = float(input('How many hours do you study a day? '))
    b = int(input('What is the percentage of your attendance? '))
    c = int(input('What is your previous test score? '))
    d = float(input('How many hours do you sleep per day? '))
    e = int(input('How many assignments have you completed this year? '))

    prdm = model.predict([[a,b,c,d,e]])

    if(prdm == 1):
        return 'The model predicts that you may pass the exam'
    else:
        return 'The model predicts that you may not pass the exam'

print(prd())


