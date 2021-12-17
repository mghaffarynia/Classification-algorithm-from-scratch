

# Primal SvM

import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def upload_data():
  from google.colab import files
  uploaded = files.upload()
  
  
def read_input(filename):
  df = pd.read_csv(filename, header = None).to_numpy()
  y = df[:, [0]]
  y = np.where(y == 0, -1, y)
  X = df[:, range(1, df.shape[1])]
  return X, y
  
def opt(X, y, c):
  m, n = X.shape
  np_P = np.identity(n + m + 1)
  for i in range(n, n+m+1):
    np_P[i, i] = 0.0
  P = matrix(np_P)
  np_q = np.ones((n + m + 1, 1)) * c
  for j in range(0, n + 1):
    np_q[j, 0] = 0.0
  q = matrix(np_q)
  temp1 = np.multiply((-1)*y, np.concatenate((X, np.ones((m, 1))), axis = 1))
  temp2 = np.concatenate((temp1, (-1)*np.ones((m, m))), axis = 1)
  temp3 = np.zeros((m, m + n + 1))
  i = 0
  for k in range(n+1, m + n + 1):
    temp3[i, k] = (-1) * 1.0
    i += 1
  np_G = np.concatenate((temp2, temp3), axis = 0)
  G = matrix(np_G)
  np_h = np.concatenate((np.ones((m, 1)) * (-1), np.ones((m, 1))), axis = 0)
  h = matrix(np_h)
  solvers.options['show_progress'] = False
  sol = solvers.qp(P, q, G, h)
  return np.array(sol['x'])
  
def solution(train_filename, val_filename, test_filename):
  X_train, y_train = read_input(train_filename)
  X_val, y_val = read_input(val_filename)
  X_test, y_test = read_input(test_filename)

  best_c = -1
  best_val_acc = -1
  for i in range(9):
    c = 10**i
    wbxi = opt(X_train, y_train, c)
    # print("w, b, Xi: ", wbxi)
    w = wbxi[range(X_train.shape[1]), :]
    b = wbxi[X_train.shape[1], :]
    train_pred = np.multiply(np.dot(X_train, w) + b, y_train)
    train_acc = np.sum(np.where(train_pred <= 0, 0, 1)) / X_train.shape[0] * 100
    val_pred = np.multiply(np.dot(X_val, w) + b, y_val)
    val_acc = np.sum(np.where(val_pred <= 0, 0, 1)) / X_val.shape[0] * 100
    if val_acc > best_val_acc:
      best_c, best_val_acc = c, val_acc
    print(f"For c={c}:")
    print(f"\tTraining accuracy:   {train_acc}%.")
    print(f"\tValidation accuracy: {val_acc}%.")
    print("*******************************************************")
  
  print(f"\nBest c={best_c}:")
  test_pred = np.multiply(np.dot(X_test, w) + b, y_test)
  test_acc = np.sum(np.where(test_pred <= 0, 0, 1)) / X_test.shape[0] * 100
  print(f"\tTest accuracy: {test_acc}%.")
  
solution("park_train.data", "park_validation.data", "park_test.data")

