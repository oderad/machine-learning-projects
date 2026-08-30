import pandas as pd
from sklearn.linear_model import LogisticRegression

scores = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\LOGISTIC_REGRESSION\exam_scores.csv')
churn = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\LOGISTIC_REGRESSION\cust_churn.csv')
loan = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\LOGISTIC_REGRESSION\loan_approval.csv')
spam = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\LOGISTIC_REGRESSION\spam_detect.csv')
weight = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\LOGISTIC_REGRESSION\weight_loss.csv')


# Scores
xs = scores[['hours_studied', 'attendance', 'previous_score']]
ys = scores['passed']

model = LogisticRegression()
model.fit(xs,ys)
pred = model.predict([[4,70,60]])

def preds():
    if (pred == 1):
        return 'Pass'
    else:
        return 'Fail'

# Customer Churn rate prediction
# Customer churning means the customer left 
xc = churn[['months_customer','monthly_bill','support_calls','contract_months']]
yc = churn['churned']

model_1 = LogisticRegression()
model_1.fit(xc, yc)

pred1 = model_1.predict([[11,55,3,11]])

def cust():
    if (pred1 == 1):
        return ' customer churned'
    else:
        return ' customer did not churn'

# Loan approval
print(loan.info())
xl = loan[['income','age','credit_score','loan_amount']]
yl = loan['approved']

model_2 = LogisticRegression()
model_2.fit(xl,yl)

loans = model_2.predict([[40000,30,630,20000]])

def lns():
    if(loans == 1):
        return 'The loan will be approved'
    else:
        return 'The loan will not be approved'

# Spam detection
print(spam.info())
print(spam)

xp = spam[['word_count','links','capital_letters', 'has_attachment']]
yp = spam['spam']

model_3 = LogisticRegression()
model_3.fit(xp,yp)

spams = model_3.predict([[90,2,25,1]])

def spm():
    if(spams == 1):
        return 'The email looks like it is spam'
    else:
        return 'The email does not look like it is spam'

# Weight loss
xw = weight[['exercise_hours', 'diet_score','sleep_hours','calories_burned']]
yw = weight['lost_weight']

model_4 = LogisticRegression()
model_4.fit(xw,yw)

weights = model_4.predict([[3,8,8,500]])

def wt():
    if(weights == 1):
        return 'the member will lose weight'
    else:
        return 'the member will not lose weight'

# Prediction Outputs
print(f'The model predicts students that studied approx 4 hours, 70% attendance, and a previous score of 60 will {preds()}')
print(f'The {cust()} with 11 months, monthly billed 55, 3 support calls, and 11 contract months')
print(f'{lns()} for someone whos annual income is $40000, age 30, 630 credit score, applying for a 20000 loan')
print(f'{spm()}')
print(f'The prediction shows that {wt()}')