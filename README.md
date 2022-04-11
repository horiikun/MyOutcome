# MyOutcome
私が卒業研究で作成したソースコードをまとめています．<br>
## テーマ<br>
### 「ゲーム木探索における深層学習を利用した評価関数の実装」

## 概要
　ボードゲームを題材に探索効率を向上させるアルゴリズムの開発を検討しました．
具体的には五目並べを対象に，既存の探索アルゴリズムを実装し，その性質を比較する．その結果をもとに探索精度が高く実時間で探索し終える評価関数を深層学習を利用して実装し，有効かどうかを検証しました．<br>　
以下に本研究で作成したソースコードの概要を記載します．

### **game.py**
　五目並べのゲームを実装したプログラム．五目並べのルールや勝敗判定の他，実験のために実装した探索アルゴリズムを記載しています．
### **NeuralNet.py**
  評価関数を実装するための学習モデルを生成したプログラム．このプログラムではシンプルなニューラルネットワークを採用したのでゲームの複雑な盤面情報をうまくとらえることが出来ず学習が進みませんでした．
### **cnn.py**
  NeuralNet.pyでとらえられなかった複雑な盤面情報をとらえるために実装した深層学習．Neuralnet.pyでは盤面の駒は自分と対戦相手の区別ができませんでしたが，畳み込みニューラルネットワークで学習させることで自分と対戦相手の区別が可能となりました．
### **self_play.py**
　学習用のデータや学習モデルの精度を確かめるテスト用のデータを生成するために自己対戦させたプログラムです．
### **check_boarddata.py**
　学習させるための盤面を生成させるプログラムです．自分が置いた手なのか，相手が置いた手なのかを区別させるアルゴリズムを実装し，主にcnn.pyで使われました．
### **check_learn.py**
　評価関数の精度を確認するためのプログラムです．学習用データとテスト用データの各正答率を確認し，評価関数の精度を考察しました．
### **data_write_smp.py**
　本研究で利用するデータや学習モデルを生成や保存，呼び出したりするためのメソッドを集めたプログラムです．
### **value_check.py**
　実装した評価関数を利用し，ゲームをするためのプログラムです．
