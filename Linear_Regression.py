# C:\Users\omard\Desktop\Educational__Information\Data_Science_Projects\Data_Analysis\Python\Machine_Learning\houses.csv
import pandas as pd
from sklearn.linear_model import LinearRegression

car = pd.read_csv(r'C:\Users\car_p.csv')

stu = pd.read_csv(r'C:\Users\study_p.csv')

hou = pd.read_csv(r'C:\Users\houses.csv')

# Check data
print(car.info())
print(stu.info())
print(hou.info())

# Car, need to predict a price
xc = car[['mileage','engine_size','age_years','horsepower']]
yc = car['price']

model_1 = LinearRegression()
model_1.fit(xc,yc)

predc = model_1.predict([[40000,2.0,5,200]])

#  Exam, need to predict an exam score
xs = stu[['hours_studied']]
ys = stu['exam_score']

model_2 = LinearRegression()
model_2.fit(xs,ys)

scr = model_2.predict([[13]])
# predicting the housing price

xh = hou[['bedrooms','bathrooms','sqft','age','location']]
yh = hou['price']

model_3 = LinearRegression()
model_3.fit(xh,yh)

predh = model_3.predict([[4, 3, 2000, 10, 1]])
# Obtain predictions based on parameters

print(f'The model predicts house with 4 bedrooms, 3 bathrooms, 2000 sqft, 10 years, in a good neighborhood is approx ${int(predh[0])}')

print(f'The model predicts a car with 40,000 mileage, 2.0 engine size, 5 years old, and 200 horsepower is approx ${int(predc[0])}')

print(f'The model predicts that when students who study around 13 hours a week have an average score of {int(scr[0])}')
