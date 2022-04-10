#パッケージのインポート
from game import State
from game import alpha_beta_action
from game import random_action
from game import alpha_beta
from game import judge
from datetime import datetime
import numpy as np
import self_play as sp
from tensorflow.keras.layers import Activation, Dense, Dropout
# from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path
import os
from data_write_smp import load_data
from check_boarddata import create_data

RCOUNTER = 20
GCOUNTER = 1
ONESIDE = 7
BOARDSIZE = ONESIDE*ONESIDE
SELFPLAY_SIZE = 10000
SIZE = 50000

#モデルの読み込み
# path = sorted(Path('./model').glob('*.h5'))[-1]
model = load_model('./model/cnnmodel.h5')

check_board = []
check_value = []

#データセットの準備.
k = []
# history_another = load_data()
# history_another = np.array(history_another)
# c = history_another[0]
# d = history_another[1]
# for i in range(SIZE):
#     a = create_data(c[i])
#     k.append(a)
#     if d[i] == -1:
#         d[i] = 2
# (train_images, train_labels) = (k, d)

a, b = sp.self_play()
for i in range(SELFPLAY_SIZE):
    c = create_data(a[i])
    k.append(c)
    if b[i] == -1:
        b[i] = 2

(train_images, train_labels) = (k, b)

#
#     check_board.append(a)
#     check_value.append(b)

check_board = np.array(train_images)
print(check_board.shape)
check_board = check_board.transpose(0, 2, 3, 1)
# check_board = check_board.reshape((check_board.shape[0], 49))
check_value = train_labels

# print(check_board)

# print(type(a))

# c = a.flatten()

# print(c)

predictions = model.predict(check_board)
# print(predictions)
predictions = np.argmax(predictions, axis=1)

count = 0

for i in range(SELFPLAY_SIZE):
    if predictions[i] == check_value[i]:
        count += 1

ASR = count / SELFPLAY_SIZE

print('正答率 = ' + str(ASR))

# print(b)
# print(predictions)
# print(check_value)
