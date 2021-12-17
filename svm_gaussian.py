

# Dual SVM
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def read_input(filename):
# Reading the data and converting it to numpy array
  df = pd.read_csv(filename, header = None).to_numpy()
# The label's column is the first column.
  y = df[:, [0]]
# As it is a binary classification, we change the value of the output to y = -1 everywhere y is zero.
  y = np.where(y == 0, -1, y)
# Feature's columns start from column 1.
  X = df[:, range(1, df.shape[1])]
  return X, y

# Writing a Function for printing the accuracy for different squared sigmas.
def print_accs(accs):
  print("                       sq_sigma")
  print(" "*9, end='')
  for j in range(5):
    print(f"{0.1*(10**j):>7.1f} ", end='')
  print()
  for i, l in enumerate(accs):
    print(f"c: 10^{i:<1d}: ", end='')
    for acc in l:
      print(f"{acc:>7.03f}", end=' ')
    print()
  print()

# Computing the Gaussian_Kernel of X and Z matices.
def Gaussian_Kernel(X, Z, sq_sigma):
  xx = np.dot(X, X.T)
  zz = np.dot(Z, Z.T)
  xz = np.dot(X, Z.T)
  norm_matrix = ((2*xz)-np.diag(xx).reshape((-1, 1))-np.diag(zz.T).reshape((1, -1)))/(2*sq_sigma)
  kernel = np.exp(norm_matrix)
  return kernel

# Computing the parameters (W)
def estimate(kernel, y_train, lambdas, b):
  est = np.sum((lambdas * y_train) * kernel, axis = 0) + b
  return est.reshape(-1, 1)
# Computing the parameter b
def calculate_b(lambdas, y, estimates, threshold):
  b = 0
  counter = 0
  for i in range(y.shape[0]):
    if lambdas[i][0] > threshold:
      b = b + (y[i][0] - estimates[i][0])
      counter += 1
  b = b / counter
  return b

# computing the accuracy of the algorithm.
def accuracy(estimates, y):
  temp = np.multiply(estimates, y)
  num = y[np.where(temp > 0)[0]]
  acc_percent = num.shape[0]/y.shape[0]
  return acc_percent * 100

# Implementing the optimization problem.
def opt(X, y, c, sq_sigma):
  m,n = X.shape
  yy = np.dot(y, y.T)
  k = Gaussian_Kernel(X, X, sq_sigma)
  H = np.multiply(k, yy)
  P = matrix(H)
  np_q = np.ones((m, 1)) * (-1.0)
  q = matrix(np_q)
  temp1 = np.identity(m) * (-1.0)
  temp2 = np.identity(m)
  np_G = np.concatenate((temp1, temp2), axis = 0)
  G = matrix(np_G)
  np_h = np.concatenate((np.zeros((m, 1)) , np.ones((m, 1)) * c), axis = 0)
  h = matrix(np_h)
  A = matrix(y.T * 1.0)
  b = matrix(np.zeros(1))

  solvers.options['show_progress'] = False
  sol = solvers.qp(P, q, G, h, A, b)
  lambdas = np.array(sol['x'])
  return lambdas

class Solution():
  def __init__(self, train_filename, val_filename):
    self.X_train, self.y_train = read_input(train_filename)
    self.X_valid, self.y_valid = read_input(val_filename)

  def train_models(self, c_list, sq_sigma_list, b_threshold = 1e-6):
    accs_train, accs_valid = [[] for c in c_list], [[] for c in c_list]
    all_lambdas, all_b = [[] for c in c_list], [[] for c in c_list]
    for i, c in enumerate(c_list):
      for sq_sigma in sq_sigma_list:
        lambdas = opt(self.X_train, self.y_train, c, sq_sigma)
        kernel_train = Gaussian_Kernel(self.X_train, self.X_train, sq_sigma)

        estimates_train = estimate(kernel_train, self.y_train, lambdas, 0)
        b = calculate_b(lambdas, self.y_train, estimates_train, b_threshold)

        estimates_train = estimate(kernel_train, self.y_train, lambdas, b)
        acc_train = accuracy(estimates_train, self.y_train)
        accs_train[i].append(acc_train)

        kernel_valid = Gaussian_Kernel(self.X_train, self.X_valid, sq_sigma)

        estimates_valid = estimate(kernel_valid, self.y_train, lambdas, b)
        acc_valid = accuracy(estimates_valid, self.y_valid)
        accs_valid[i].append(acc_valid)
        all_lambdas[i].append(lambdas)
        all_b[i].append(b)
    print("Accuracies for train: ")
    print_accs(accs_train)
    print("Accuracies for valid: ")
    print_accs(accs_valid)
    self.accs_train, self.accs_valid = np.array(accs_train), np.array(accs_valid)
    best_c_idx, best_sq_sigma_idx = np.unravel_index(
        np.argmax(self.accs_valid, axis=None), self.accs_valid.shape)
    self.best_lambdas = all_lambdas[best_c_idx][best_sq_sigma_idx]
    self.best_b = all_b[best_c_idx][best_sq_sigma_idx]
    self.best_c, self.best_sq_sigma = 10**best_c_idx, 10.0**(best_sq_sigma_idx-1)
    print(f"best c:{self.best_c}\nsq_sigma:{self.best_sq_sigma}\n")

# Running the algorithm on the test data.
  def test(self, X_test, y_test):
    kernel_test = Gaussian_Kernel(self.X_train, X_test, self.best_sq_sigma)
    estimates_test = estimate(
        kernel_test, self.y_train, self.best_lambdas, self.best_b)
    acc_test = accuracy(estimates_test, y_test)
    return acc_test, estimates_test

sol = Solution("park_train.data", "park_validation.data")

# Different regularization parameters as hyperparameter.
c_list = [10**i for i in range(5)]
# Different squared sigma as hyperparameter.
sq_sigma_list = [(10.0**(j-1)) for j in range(5)]
sol.train_models(c_list, sq_sigma_list)

X_test, y_test = read_input("park_test.data")
acc_test, estimates_test = sol.test(X_test, y_test)

print("Test accuracy: ", acc_test)