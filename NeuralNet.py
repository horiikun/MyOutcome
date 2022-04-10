# coding: UTF-8

from self_play import self_play
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from data_write_smp import load_data
import numpy as np
import matplotlib.pyplot as plt
import os

SIZE = 50000

#データセットの準備.
k = []
history_another = load_data()
history_another = np.array(history_another)
c = history_another[0]
for i in range(SIZE):
    k.append(c[i])
(train_images, train_labels) = (k, history_another[1])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# (train_images, train_labels) = self_play()

#データセットのシェイプの確認
print(train_images.shape)
print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# #データセットの画像の確認
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(train_images[i], 'gray')
# plt.show()
#
# #データセットのラベルの確認
# print(train_labels[0:10])

#データセットの画像の前処理
train_images = train_images.reshape((train_images.shape[0], 49))
# test_images = test_images.reshape((test_images.shape[0], 784))

#データセットの画像の前処理後のシェイプの確認
# print(train_images.shape)
# print(test_images.shape)

#データセットのラベルの前処理
train_labels = to_categorical(train_labels, 3)
# test_labels = to_categorical(test_labels)

# #データセットのラベルの前処理後のシェイプの確認
print(train_labels.shape)
# print(test_labels.shape)

#モデルの生成
model = Sequential()
model.add(Dense(256, activation='sigmoid', input_shape=(49,))) #入力層
model.add(Dense(128, activation='sigmoid')) #隠れ層
model.add(Dropout(rate=0.5)) #ドロップアウト
model.add(Dense(3, activation='softmax')) #出力層

#コンパイル
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])

#学習
history = model.fit(train_images, train_labels, batch_size=128, epochs=50, validation_split=0.1)

#モデルの保存
os.makedirs('./model', exist_ok=True)
model.save('./model/model.h5')

#グラフの表示
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

#評価
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc ))

#推論する画像の表示
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(test_images[i].reshape((28, 28)), 'gray')
# plt.show()

#推論したラベルの表示
# test_predictions = model.predict(test_images[0:10])
# test_predictions = np.argmax(test_predictions, axis=1)
# print(test_predictions)

print('終了しました')
