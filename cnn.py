# coding: UTF-8

#パッケージのインポート
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from data_write_smp import load_data
from check_boarddata import create_data
import os

SIZE = 50000

#データセットの準備.
k = []
history_another = load_data()
history_another = np.array(history_another)
c = history_another[0]
d = history_another[1]
# print(c)
for i in range(SIZE):
    a = create_data(c[i])
    k.append(a)
    if d[i] == -1:
        d[i] = 2
(train_images, train_labels) = (k, d)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images = train_images.transpose(0, 2, 3, 1)

print(train_images.shape)
print(train_labels.shape)

#データセットのラベルの確認
# print(train_labels[0:10])

#データセットの画像の前処理
# train_images = train_images.reshape((train_images.shape[0], 49))

#データセットの画像の前処理後のシェイプの確認
print(train_images.shape)

#データセットのラベルの前処理
train_labels = to_categorical(train_labels, 3)

#データセットのラベルの前処理後のシェイプの確認
print(train_labels.shape)

#モデルの生成
model = Sequential()

#Conv→Conv→Pool→Dropout
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Conv→Conv→Pool→Dropout
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Dense→Dropout→Dense
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

#コンパイル
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

#学習
history = model.fit(train_images, train_labels, batch_size=500, epochs=50, validation_split=0.1)

#モデルの保存
os.makedirs('./model', exist_ok=True)
model.save('./model/cnnmodel.h5')

#グラフの表示
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

print('終了しました')
