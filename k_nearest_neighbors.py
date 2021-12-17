

# K-nearest neighbours

import pandas as pd
import numpy as np

def upload_data():
  from google.colab import files
  uploaded = files.upload()

def read_input(filename):
  df = pd.read_csv(filename, header = None).to_numpy()
  y = df[:, [0]]
  y = 2*y-1
  X = df[:, range(1, df.shape[1])]
  return X, y

def normalization(X, mean, std):
  return (X - mean) / std

def distance(X1, X2):
  X1_sq = np.diag(np.dot(X1, X1.T)).reshape(X1.shape[0], -1)
  X2_sq = np.diag(np.dot(X2, X2.T)).reshape(-1, X2.shape[0])
  X1X2_2 = 2*np.dot(X1, X2.T)
  dist = X1_sq + X2_sq - X1X2_2
  return np.sqrt(dist)

def knn_accuracy(distance_matrix, k, y_train, y):
  _, m2 = distance_matrix.shape
  index_sort = np.argsort(distance_matrix, axis=0)
  kny = np.take_along_axis(y_train, index_sort[0:k, :], axis=0)
  predicts = np.sum(kny, axis=0).reshape(-1, m2)
  accs = np.sum(np.where((predicts*y.T) < 0, 0, 1))
  accuracy = (accs / m2) * 100
  return accuracy

def solution(train_filename, val_filename, test_filename, list_of_k):
  X_train, y_train = read_input(train_filename)
  X_val, y_val = read_input(val_filename)
  X_test, y_test = read_input(val_filename)
  _, n = X_train.shape
  mean = np.mean(X_train, axis = 0).reshape(-1, n)
  std = np.std(X_train, axis=0).reshape(-1, n)
  X_train = normalization(X_train, mean, std)

  X_val = normalization(X_val, mean, std)
  dis_X_train = distance(X_train, X_train)
  dis_X_val = distance(X_train, X_val)
  dis_X_test = distance(X_train, X_test)

  best_k = -1
  best_val_acc = -1
  for k in list_of_k:
    print(f"For k={k}:")
    train_accuracy = knn_accuracy(dis_X_train, k, y_train, y_train)
    print(f"\tTraining accuracy is :   {train_accuracy}")
    val_accuracy = knn_accuracy(dis_X_val, k, y_train, y_val)
    print(f"\tValidation accuracy is : {val_accuracy}")
    if val_accuracy > best_val_acc:
      best_k, best_val_acc = k, val_accuracy
    print("#####################################################")

  print(f"\nBest k in validation = {best_k}")
  test_accuracy = knn_accuracy(dis_X_test, k, y_train, y_test)
  print(f"\tTest accuracy is : {test_accuracy}")


list_of_k = [1, 5, 11, 15, 21]
solution("park_train.data", "park_validation.data", "park_test.data", list_of_k)
