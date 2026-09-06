import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

scr = pd.read_csv(r'C:\Users\scores.csv')

# Student Test scores
scr['Pass_Measure'] = scr['Pass'].str.replace({'Yes':'1','No':'0'})
scr['Pass_Measure'] = pd.to_numeric(scr['Pass_Measure'],errors='coerce')

xs = scr[['StudyHours','Attendance']]
ys = scr['Pass_Measure']

# Split the data for training
xs_test, xs_train, ys_test, ys_train = train_test_split(
    xs,
    ys,
    test_size = 0.45,
    random_state = 42
)

# Create and train the model
model = DecisionTreeClassifier(random_state = 42)
model.fit(xs_train, ys_train)

# Predict the model on the unseen data
# The xs_test, values the model did not see yet so here is where it will predict using the remaining x values
# Note, these are the input x values to test
predictions = model.predict(xs_test)

# Accuracy
# This compares the remaing y results with the remaining x results to view accuracy
accuracy = accuracy_score(ys_test, predictions)

print(accuracy)
