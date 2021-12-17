
import pandas as pd
import numpy as np


def read_input(filename):
  df = pd.read_csv(filename, header = None).to_numpy()
  y = df[:, 0]*2-1
  X = df[:, range(1, df.shape[1])]
  return X, y

def normalize(X):
  means = np.mean(X, axis=0)
  stds = np.std(X, axis=0)
  return (X-means)/stds, means, stds

def prob(X, w, b):
  return 1-(1/(1+np.exp(np.dot(X, w)+b)))

def lr_log_loss(X, y, w, b, lam=0, reg_type=''):
  probs = prob(X, w, b)
  if reg_type == 'ridge':
    reg = np.sum(w**2)*(lam/2)
  elif reg_type == 'lasso':
    reg = np.sum(np.absolute(w))*lam
  else:
    reg = 0
  log_loss = np.sum(np.log(probs)*(y == 1) + np.log(1-probs)*(y == -1)) - reg
  return log_loss

def lr_derivative(X, y, w, b, lam=0, reg_type=''):
  probs = prob(X, w, b)
  if reg_type == 'ridge':
    dreg = w*lam
  elif reg_type == 'lasso':
    dreg = ((w > 0)*2-1)*lam
  else:
    dreg = 0
  dw = np.sum(((y == 1)-probs).reshape(-1, 1)*X, axis=0) - dreg
  db = np.sum((y == 1)-probs)
  return dw, db

def logestic_regression(X, y, lam=0, reg_type=''):
  m, n = X.shape
  w, b = np.zeros(n), 0
  step_size = 1e-5
  losses = []
  while True:
    dw, db = lr_derivative(X, y, w, b, lam, reg_type)
    w += (step_size*dw)
    b += (step_size*db)
    loss = lr_log_loss(X, y, w, b, lam, reg_type)
    losses.append(loss)
    if (len(losses) > 2 and abs(loss - losses[-2]) < 1e-3):
      break
  return w, b, losses

def calc_accuracy(X, y, w, b):
  m, _ = X.shape
  probs = prob(X, w, b)
  y_hat = (probs > 0.5)*2-1
  return np.sum(y == y_hat)/m*100

def print_accs(accs, lams):
  print(f"Accuracy with no regularization: {accs[0][0]:.2f}%\n")

  print(f"Accuracies with l2 regularization:")
  lams_str = ' '.join([f"{lam:7.3f}" for lam in lams])
  print(f"lambda: {lams_str}")
  accs_str = ' '.join([f"{acc:6.2f}%" for acc in accs[1]])
  print(f"accs:   {accs_str}\n")

  print(f"Accuracies with l1 regularization:")
  lams_str = ' '.join([f"{lam:7.3f}" for lam in lams])
  print(f"lambda: {lams_str}")
  accs_str = ' '.join([f"{acc:6.2f}%" for acc in accs[2]])
  print(f"accs:   {accs_str}")
  print("#########################################")
  print("\n\n")

X, y = read_input('park_train.data')
X_norm, means, std = normalize(X)
X_test, y_test = read_input('park_test.data')
X_test = (X_test-means)/std
X_val, y_val = read_input('park_validation.data')
X_val = (X_val-means)/std

lams = [0.01, 0.1, 1, 10, 100]
train_accs, val_accs, test_accs = [], [], []
for reg_type in ['', 'ridge', 'lasso']:
  train_reg_accs, val_reg_accs, test_reg_accs = [], [], []
  for lam in (lams if reg_type != '' else [0]):
    w, b, losses = logestic_regression(X_norm, y, lam, reg_type)
    train_acc = calc_accuracy(X_norm, y, w, b)
    val_acc = calc_accuracy(X_val, y_val, w, b)
    test_acc = calc_accuracy(X_test, y_test, w, b)
    train_reg_accs.append(train_acc)
    val_reg_accs.append(val_acc)
    test_reg_accs.append(test_acc)
  train_accs.append(train_reg_accs)
  val_accs.append(val_reg_accs)
  test_accs.append(test_reg_accs)

print("Train accuracies: ")
print_accs(train_accs, lams)
print("val accuracies: ")
print_accs(val_accs, lams)
print("test accuracies: ")
print_accs(test_accs, lams)