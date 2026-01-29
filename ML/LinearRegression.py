import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def call():
  np.random.seed(42)
  x = np.random.rand(50,1)*100
  y = 3.5*x + np.random.rand(50,1)*20
  #print(x)
  #print(y)
  model = LinearRegression()
  model.fit(x,y)
  Y_pred = model.predict(x)
  #print(Y_pred)
 
  plt.figure(figsize=(8, 6))
  plt.scatter(x, y, color='blue', label='Data Points')
  plt.plot(x, Y_pred, color='red', linewidth=2, label='Regression Line')
  plt.title('Linear Regression on Random Dataset')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()
  plt.grid(True)
  plt.show()

if  __name__ == "__main__":
    call()