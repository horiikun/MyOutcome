# coding: UTF-8

#パッケージのインポート
from game import State
from game import alpha_beta_action
from game import random_action
from game import alpha_beta
from game import judge
from datetime import datetime
from data_write_smp import write_data
from data_write_smp import load_data
from pathlib import Path
import numpy as np
import pickle
import os

RCOUNTER = 20
GCOUNTER = 1
ONESIDE = 7
BOARDSIZE = ONESIDE*ONESIDE
SELFPLAY_SIZE = 10000

#先手プレーヤーの価値
def first_player_value(ended_state):
    #1:先手勝利,-1:先手敗北,0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

#1次元配列のデータを2次元配列に変換
def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]

#ゲームの状態を一つにまとめる
def one_bordstate(a, b):
    bs = [[0]*ONESIDE for i in range(ONESIDE)]

    for i in range(ONESIDE):
        for j in range(ONESIDE):
            if a[i][j] == 1:
                bs[i][j] = 1
            elif b[i][j] == 1:
                bs[i][j] = 2
            else:
                bs[i][j] = 0
    return bs

#ゲームの状態の評価値を決定
def GameValue(values):
    Vsum = 0
    v = 0
    Vsum = sum(values)
    # print(Vsum)
    if Vsum > 0:
        v = 1
    elif Vsum < 0:
        v = -1
    else:
        v = 0
    return v

#1ゲームの実行
def play():
    #学習データ
    # history = []
    gcount = 0
    values = []
    gc = 0
    boardstate = np.arange(BOARDSIZE)
    boardstateE = []
    bs = [[0]*ONESIDE for i in range(ONESIDE)]
    score = 0

    while True:
        if gc > GCOUNTER-1:
            # print(gc)
            break
        #状態の生成
        state = State()
        count = 0
        #ゲーム終了までのループ
        while True:
            #ゲーム終了時
            if state.is_done():
                # judge(state)
                break

            #行動の取得
            if state.is_first_player():
                count+=1
                if count > RCOUNTER:
                    boardstate = state.pieces
                    boardstateE = state.enemy_pieces
                    score = playdata(state)
                    gc += 1
                    # print(score)
                    break
                else:
                    action = random_action(state)
            else:
                action = random_action(state)

            #次の状態の取得
            state = state.next(action)

    # print('これです↓')
    # print(boardstate)
    boardstate = convert_1d_to_2d(boardstate, ONESIDE)
    boardstateE = convert_1d_to_2d(boardstateE, ONESIDE)
    bs = one_bordstate(boardstate, boardstateE)
    # print(np.ndim(bs))
    # print(boardstate)
    # print(boardstateE)
    # print(bs)
    # print(score)
    # print(v)

    return bs, score

#alphabetaの探索
def playdata(state):
    score = 0
    alpha = -float('inf')
    # print(state.pieces)
    #評価値の取得
    score = alpha_beta(state, -float('inf'), -alpha)
    return score


#セルフプレイ
def self_play():
    train_data = []
    train_label = []
    count = 0
    for _ in range(SELFPLAY_SIZE):
        a, b = play()
        train_data.append(a)
        train_label.append(b)
        count += 1
        print('\r試行回数:{:4d}/{:4d}'.format(count, SELFPLAY_SIZE), end='')
    print()
    # train_data = np.array(train_data)
    # print(train_data)
    # print(train_data.ndim)
    # train_label = np.array(train_label)
    return train_data, train_label



#動作確認
# if __name__ == '__main__':
#     history = []
#     a, b = self_play()
#     history = [a, b]
#     # print(history)
#     # print(len(history))
#     # print(type(history))
#     history = np.array(history)
#     # print(history)
#     # print(history.ndim)
#     write_data(history)
#     print('データを書き込みました')
    # history_another = load_data()
    # print(history_another)
    # print('\n')
    # (board_st, v) = (history_another[0], history_another[1])
    # board_st = history_another[0]
    # v = history_another[1]
    # print(board_st)
    # print('\n')
    # print(v)
    # board_st = np.array(board_st)
    # print(board_st)
    # print(board_st.ndim)
    # k = []
    # print(board_st[0])
    # h = np.array(board_st[0])
    # print(h)
    # print(h.ndim)
    # for _ in range(10):

    # board_st = np.array(board_st)
    # print(board_st.ndim)
    # print(v)
