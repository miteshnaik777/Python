import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def call():
  X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
  Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

  model = LogisticRegression()
  model.fit(X,Y)
  prdict = model.predict(X)
  print("Predicted classes:", prdict)


  X_test = np.linspace(0, 9, 100).reshape(-1, 1)
  y_prob = model.predict_proba(X_test)[:, 1]

  plt.scatter(X, Y, color='red', label='Actual data')
  plt.plot(X_test, y_prob, color='blue', label='Sigmoid curve')
  plt.xlabel("Hours Studied")
  plt.ylabel("Probability of Passing")
  plt.legend()
  plt.show()

if  __name__ == "__main__":
    call()