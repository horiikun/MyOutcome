# coding: UTF-8

from datetime import datetime
from pathlib import Path
import pickle
import os

def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) #フォルダが無いときは生成
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history,f)

def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    # print(history_path)
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    a = [0, 1, 2]
    write_data(a)
    print('データを書き込みました')

    history = load_data()

    print(history)
