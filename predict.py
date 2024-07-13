import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('heart_v2.csv', delimiter=',')
df.dataframeName = 'heart_v2.csv'

# Prepare the data
x = df.drop('heart disease', axis=1)
y = df['heart disease']

# Load the saved model and GridSearchCV object
grid_search = joblib.load('grid_search.pkl')
rf_best = joblib.load('rf_best.pkl')

# Make predictions
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

best_accuracy = rf_best.score(x_test, y_test)

print(f"Best accuracy on test data is {round(best_accuracy*100, 2)}% with the following parameters:")
print('Best estimator :', grid_search.best_estimator_)
print('Best parameters :', grid_search.best_params_)

# Optionally, plot the decision tree (if desired)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Showing the some of the decision trees
for i in range(0, 5):
    plt.figure(figsize=(10, 5))
    # feature = (lambda x: grid_search.feature_names_in_[x] if (x<4) else 'disease')(i)
    plot_tree(rf_best[i], class_names=['disease', 'no disease'], filled=True, proportion=True)
    plt.show()
