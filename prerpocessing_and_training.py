import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('heart_v2.csv', delimiter=',')
df.dataframeName = 'heart_v2.csv'

print(df.head(10))

# Prepare the data
x = df.drop('heart disease', axis=1)
y = df['heart disease']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Train the RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
classifier_rf.fit(x_train, y_train)

print('oob score =', classifier_rf.oob_score_)

# Grid search for best parameters
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 15, 20, 50, 100, 200],
    'n_estimators': [10, 25, 30, 50, 100, 200]
}

grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Save the best model and GridSearchCV object
joblib.dump(grid_search, 'grid_search.pkl')
joblib.dump(grid_search.best_estimator_, 'rf_best.pkl')

train_accuracy = grid_search.best_score_

print(f"Best accuracy on the training dataset : {round(train_accuracy*100, 2)}%")