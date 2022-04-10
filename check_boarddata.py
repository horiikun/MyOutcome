from data_write_smp import load_data
from self_play import self_play
from self_play import convert_1d_to_2d
from tensorflow.keras.utils import to_categorical
import numpy as np

ONESIDE = 7
SIZE = 50000

def create_data(train_data):
    board = []
    board_e = []
    data = []
    k = []

    # print(train_images)
    #自分が置いた手なのか相手が置いた手なのかを判定
    train_images = np.array(train_data)
    a = train_images
    for i in range(7):
        for j in range(7):
            if a[i][j] == 0:
                board.append(0)
                board_e.append(0)
            elif a[i][j] == 1:
                board.append(1)
                board_e.append(0)
            else:
                board.append(0)
                board_e.append(1)



    board = convert_1d_to_2d(board, ONESIDE)
    board = np.array(board)
    board_e = convert_1d_to_2d(board_e, ONESIDE)


    for i in range(len(board)):
        for j in range(len(board)):
            data.append(board[i][j])
    for i in range(len(board_e)):
        for j in range(len(board_e)):
            data.append(board_e[i][j])

    data = np.array(data)
    # print(data)
    data = np.reshape(data, (2, 7, 7))
    # print(data.shape)
    # print(data)
    return data

if __name__ == '__main__':

    #データセットの準備.
    k = []
    history_another = load_data()
    history_another = np.array(history_another)
    c = history_another[0]
    train_labels = history_another[1]
    # print(c)
    # print(c[0])
    # for i in range(SIZE):
    #     a = create_data(c[i])
    #     k.append(a)
    # (train_images, train_labels) = (k, history_another[1])

    # train_images = np.array(train_images)
    for i in range(SIZE):
        if train_labels[i] == -1:
            train_labels[i] = 2

    train_labels = np.array(train_labels)

    print(train_labels[0:10])

    train_labels = to_categorical(train_labels, 3)

    # train_images = train_images.transpose(0, 2, 3, 1)

    # print(train_images.shape)
    print(train_labels.shape)

    print(train_labels[0:10])
