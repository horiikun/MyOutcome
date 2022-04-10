# coding: UTF-8

#五目並べの作成
import random
import math
import numpy as np
from termcolor import colored, cprint
import colorama
import time
import csv

RCOUNTER = 20
ONESIDE = 7
BOARDSIZE = ONESIDE * ONESIDE
GCOUNTER = 1

#ゲームの状態
class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        #石の配置
        self.pieces = pieces if pieces != None else [0] * BOARDSIZE
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * BOARDSIZE

    #石の数の取得
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    #負けかどうか
    def is_lose(self):
        #5並びかどうか
        def is_comp(x, y, dx, dy):
            for k in range(5):
                if y < 0 or ONESIDE-1 < y or x < 0 or ONESIDE-1 < x or self.enemy_pieces[x+y*ONESIDE] == 0:
                    return False
                x, y = x+dx, y+dy
            return True

        #負けかどうか
        for i in range(ONESIDE):
            for j in range(ONESIDE):
                if is_comp(i, j, 1, 1) or is_comp(i, j+4, 1, -1):
                    return True
        for i in range(ONESIDE):
            for j in range(ONESIDE):
                if is_comp(i, j, 1, 0) or is_comp(i, j, 0, 1):
                    return True
        return False

    #引き分けかどうか
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == BOARDSIZE

    #ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    #次の状態の取得
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    #合法手のリストの取得
    def legal_actions(self):
        actions = []
        for i in range(BOARDSIZE):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def make_state(self):
        st_list = []
        for i in range(BOARDSIZE):
            if self.pieces[i] == 1:
                st_list.append(1)
            elif self.enemy_pieces[i] == 1:
                st_list.append(2)
            else:
                st_list.append(0)
        return st_list

    #先手かどうか
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    #文字列表示
    def __str__(self):
        colorama.init()
        o = colored('o', 'red')
        ox = (o, 'x') if self.is_first_player() else ('x', o)
        str = ''
        for i in range(BOARDSIZE):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '_'
            if i % ONESIDE == ONESIDE-1:
                str += '\n'
        return str


#ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

#ミニマックス法で状態価値計算
def mini_max(state):
    #負けは状態価値-1
    if state.is_lose():
        return -1
    #引き分けは状態価値0
    if state.is_draw():
        return 0

    #合法手の状態価値の計算
    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_score = score

    return best_score

#ミニマックス法で行動選択
def mini_max_action(state):
    #合法手の状態価値の計算
    best_action = 0
    best_score = -float('inf')
    str = ['','']
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_action = action
            best_score = score

        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('action:', str[0], '\nscore:', str[1], '\n')

    #合法手の状態価値の最大値を持つ行動を返す
    return best_action

#アルファベータ法で状態価値計算
def alpha_beta(state, alpha, beta):
    #count += 1
    #print(count)

    #負けは状態価値:-1
    if state.is_lose():
        return -1
    #引き分けは状態価値:0
    if state.is_draw():
        return 0

    #print('LegalactionSize = ' + str(a))
    #合法手の状態価値の計算
    for action in state.legal_actions():
        #makes = state.make_state()
        #for i in range(has):
            #if st_list[i] == makes:
                #print(i)
                #print(alpha)
                #return alpha
        #st_list.append(makes)
        #has += 1
        #if a < 10:
        #print(state)
        score = -alpha_beta(state.next(action), -beta, -alpha)
        #print('a')
        #print('Score = ' + str(score))
        #print('NowAction2 = ' + str(action))
        if score > alpha:
            alpha = score
            #print('a=' + str(alpha))

        #現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            #print('b=' + str(alpha))
            return alpha

    #合法手の状態価値を返す
    #print('たちつてと')
    return alpha

#アルファベータ法で行動選択
def alpha_beta_action(state):
    #合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    str = ['','']
    #count = 0
    for action in state.legal_actions():

        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        #print('ひとし')
        if score > alpha:
            best_action = action
            alpha = score

        #str[0] = '{}{:2d},'.format(str[0], action)
        #str[1] = '{}{:2d},'.format(str[1], score)
    #print('action:', str[0], '\nscore:', str[1], '\n')
    #合法手の状態価値の最大値を持つ行動を返す
    return best_action

#プレイアウト
def playout(state):
    #負けは状態価値:-1
    if state.is_lose():
        return -1

    #引き分けは状態価値:0
    if state.is_draw():
        return 0

    #次の状態の状態価値
    return -playout(state.next(random_action(state)))

#原始モンテカルロ探索で行動選択
def mcs_action(state):
    #合法手ごとに10回プレイアウトした時の状態価値の合計の計算
    legal_actions = state.legal_actions()
    values = [0] * len(legal_actions)
    for i, action in enumerate(legal_actions):
        for _ in range(10):
            values[i] += -playout(state.next(action))

    #合法手の状態価値の合計の最大値を持つ行動を返す
    return legal_actions[argmax(values)]

#最大値のインデックスを返す
def argmax(collection, key=None):
    return collection.index(max(collection))

#モンテカルロ木探索の行動選択
def mcts_action(state):
    #モンテカルロ木探索のノードの定義
    class Node:
        #初期化
        def __init__(self, state):
            self.state = state #状態
            self.w = 0 #累計価値
            self.n = 0 #試行回数
            self.child_nodes = None #子ノード群

        #局面の価値の計算
        def evaluate(self):
            #ゲーム終了時
            if self.state.is_done():
                #勝敗結果で勝ちを取得
                value = -1 if self.state.is_lose() else 0 #負けは-1,引き分けは0

                #累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            #子ノードが存在しない時
            if not self.child_nodes:
                #プレイアウトで勝ちを取得
                value = playout(self.state)

                #累計価値と試行回数の更新
                self.w += value
                self.n += 1

                #子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            #子ノードが存在するとき
            else:
                #UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                #累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        #UCB1が最大の子ノードの取得
        def next_child_node(self):
            #試行回数が0の子ノードを探す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            #UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n+(2*math.log(t)/child_node.n)**0.5)

            #UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]

    #現在の局面のノードの生成
    root_node = Node(state)
    root_node.expand()

    #100回のシミュレーションを実行
    for _ in range(100):
        root_node.evaluate()
        print('\r試行回数:{:4d}/{:4d},  評価値:{:2d}'.format(_+1, 100, root_node.evaluate()), end='')
    print()

    #試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

def judge(state):
    if state.is_first_player():
        if state.is_lose():
            # print('後手の勝ち')
            return -1
        elif state.is_draw():
            # print('引き分け')
            return 0
        else:
            # print('先手の勝ち')
            return 1
    else:
        if state.is_lose():
            # print('先手の勝ち')
            return 1
        elif state.is_draw():
            # print('引き分け')
            return 0
        else:
            # print('後手の勝ち')
            return -1


# 動作確認
if __name__ == '__main__':
    gcount = 0
    # gametime_list = []
    # log_gametime_list = []
    # ave = [0,0]
    # MAX = [0,0]
    # MIN = [0,0]
    w = []
    while True:
        #状態の生成
        state = State()
        count = 0
        tc = 0
        if gcount > GCOUNTER-1:
        #     MAX[0] = max(gametime_list)
        #     MAX[1] = max(log_gametime_list)
        #     MIN[0] = min(gametime_list)
        #     MIN[1] = min(log_gametime_list)
        #     ave[0] = sum(gametime_list)/len(gametime_list)
        #     ave[1] = math.log10(ave[0]*1000)
        #     with open('C:/Users/Owner/Documents/大学/2020卒研/dataexcel/gametimedata.csv', 'a') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(gametime_list)
        #         writer.writerow(log_gametime_list)
        #         writer.writerow(MAX)
        #         writer.writerow(MIN)
        #         writer.writerow(ave)
            break

        #ゲーム終了までのループ
        while True:
            #ゲーム終了時
            if state.is_done():
                #print('ゲーム終了')
                if count > RCOUNTER:
                    t2 = time.time()
                    elapsed_time = t2 - t1
                    a = judge(state)
                    w.append(a)
                    # gametime_list.append(elapsed_time)
                    # gg = math.log10(elapsed_time*1000)
                    # log_gametime_list.append(gg)
                    print('\r試行回数:{:4d}/{:4d}'.format(gcount, GCOUNTER), end='')
                break

            #次の状態を取得
            if state.is_first_player():
                count+=1
                if count > RCOUNTER:
                    if tc == 0:
                        t1 = time.time()
                        tc+=1

                        gcount+=1
                    action = alpha_beta_action(state)
                else:
                    action = random_action(state)
            else:
                action = random_action(state)

            state = state.next(action)

            #文字列表示
            if count == RCOUNTER:
                print(state)
                print()

    # print()
    # print('listsize=')
    # print(len(w))
    # print(w)
    # ave = sum(w) / len(w)
    # print(ave)
