
import numpy as np
from tensorflow.keras.models import load_model
from data_write_smp import load_data

RCOUNTER = 20
GCOUNTER = 1
ONESIDE = 7
BOARDSIZE = ONESIDE*ONESIDE
SELFPLAY_SIZE = 1000
SIZE = 50000

#モデルの読み込み
model = load_model('./model/model.h5')
check_value = []

lose = []
drow = []
win = []

#データセットの準備.
k = []
history_another = load_data()
history_another = np.array(history_another)
c = history_another[0]
for i in range(SIZE):
    k.append(c[i])
(train_images, train_labels) = (k, history_another[1])

check_value = train_labels

for i in range(SIZE):
    if check_value[i] == -1:
        lose.append(check_value[i])
    elif check_value[i] == 0:
        drow.append(check_value[i])
    else:
        win.append(check_value[i])

# print(lose)
print(len(lose))
# print(drow)
print(len(drow))
# print(win)
print(len(win))
