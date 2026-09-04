import pandas as pd
from sklearn.cluster import KMeans

spend = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Means\customer_spending.csv')
performance = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Means\student_performance.csv')
customers = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Means\mall.csv')
medical = pd.read_csv(r'C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\K_Means\medical.csv')

# Customer spending cluster
xs = spend[['annual_income','spending_score']]

kmeans = model = KMeans(n_clusters=3, random_state=42)
model.fit(xs)

spend['cluster'] = kmeans.fit_predict(spend[['annual_income','spending_score']])

preds = model.labels_

# Student performance
xp = performance[['Study_Hours','Exam_Score']]

model_1 = KMeans(n_clusters = 3, random_state=42)
model_1.fit(xp)

performance['cluster'] = kmeans.fit_predict(performance[['Study_Hours','Exam_Score']])

perf = model_1.labels_

# Mall customers
xc = customers[['Age','Annual_Income','Spending_Score','Visits_Per_Month']]

model_2 = KMeans(n_clusters=3, random_state=42)
model_2.fit(xc)

customers['clusters'] = kmeans.fit_predict(customers[['Age','Annual_Income','Spending_Score','Visits_Per_Month']])

predcs = model_2.labels_

# Medical
xm = medical[['Age','BMI','Systolic_BP','Cholesterol','Glucose']]

model_3 = KMeans(n_clusters = 3, random_state=42)
model_3.fit(xm)

predm = model_3.labels_

medical['clusters'] = predm

# Print your desired clustered dataset here

print(medical)
