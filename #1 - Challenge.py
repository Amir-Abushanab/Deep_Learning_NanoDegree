import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv("challenge_dataset#1.txt", names=['X','Y'])
sns.regplot(x='X', y='Y', data=data, fit_reg=False)
plt.show()

X_train, X_test, y_train, y_test = np.asarray(model_selection.train_test_split(data['X'], data['Y'], test_size=0.1))

reg = linear_model.LinearRegression()
reg.fit(X_train.values.reshape(-1,1), y_train.values.reshape(-1,1))
print('Score', reg.score(X_test.values.reshape(-1,1), y_test.values.reshape(-1,1)))