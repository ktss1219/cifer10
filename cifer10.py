import numpy as np
#import tensorflow.keras as keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing import image

# CIFAR-10データセットの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 画像データの前処理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# モデルの構築
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルのトレーニング
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# モデルの保存
model.save('cifar10_model.h5')

# トレーニング済みモデルの読み込み
model = load_model('cifar10_model.h5')

# 分類する入力画像の読み込み
img_path = 'test/鹿.jpeg' 
img = image.load_img(img_path, target_size=(32, 32))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# 予測
predictions = model.predict(img)
class_index = np.argmax(predictions)

# クラスのラベル
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class = class_labels[class_index]

print('この画像は'+predicted_class+'です。')
