import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.api.keras import optimizers, losses
from tensorflow.python.keras.api.keras.callbacks import EarlyStopping
from tensorflow.python.keras.api.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.api.keras.layers import Dense, Dropout, Input

path = 'D:\\face_recognition\\dataset'
labels = os.listdir(path)

train_inputs = []
train_outputs = []

for label in labels:
    for image in tqdm(os.listdir(os.path.join(path, label))):
        img = cv2.imread(os.path.join(path, label, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        img = img.reshape(-1)
        train_inputs.append(img)
        train_outputs.append(label)

train_inputs = np.array(train_inputs).astype('float32')/255.0
train_outputs = LabelBinarizer().fit_transform(train_outputs)

x_train_val, x_test, y_train_val, y_test = train_test_split(train_inputs, train_outputs, test_size=0.1, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1, shuffle=True)

del train_inputs, train_outputs, x_train_val, y_train_val

first_ae = Sequential()
first_ae.add(Input(shape=x_train.shape[1:]))
first_ae.add(Dense(50, activation='relu'))
first_ae.add(Dropout(0.4))
first_ae.add(Dense(x_train.shape[-1], activation='relu'))

epochs = 20
steps_per_epoch = len(x_train)/epochs

first_ae.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
history_first_ae = first_ae.fit(x_train, x_train,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=(x_val, x_val),
                                verbose=2)

plt.plot(history_first_ae.history['loss'],  label='train')
plt.plot(history_first_ae.history['val_loss'], label='validation')
plt.legend()
plt.show()

second_ae_inputs_train = first_ae.layers[0](x_train).numpy()
second_ae_inputs_val = first_ae.layers[0](x_val).numpy()

second_ae = Sequential()
second_ae.add(Input(shape=second_ae_inputs_train.shape[1:]))
second_ae.add(Dense(10, activation='relu'))
second_ae.add(Dropout(0.4))
second_ae.add(Dense(second_ae_inputs_train.shape[-1], activation='relu'))

epochs = 50
batch_size = 32

second_ae.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
history_second_ae = second_ae.fit(second_ae_inputs_train, second_ae_inputs_train,
                                  batch_size=batch_size, epochs=epochs,
                                  validation_data=(second_ae_inputs_val, second_ae_inputs_val),
                                  verbose=2)

plt.plot(history_second_ae.history['loss'],  label='train')
plt.plot(history_second_ae.history['val_loss'], label='validation')
plt.legend()
plt.show()

final_autoencoder = Sequential()

final_autoencoder.add(Input(shape=x_train.shape[1:]))
final_autoencoder.add(first_ae.layers[0])
final_autoencoder.add(second_ae.layers[0])
final_autoencoder.add(second_ae.layers[2])
final_autoencoder.add(Dense(8, activation='sigmoid'))

for layer in final_autoencoder.layers[:-1]:
    layer.trainable = False

print(final_autoencoder.summary())

eagerly_stopping = EarlyStopping(patience=5, mode='min', monitor='val_loss', restore_best_weights=True)

sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9)
final_autoencoder.compile(optimizer=sgd, loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

epochs=200
batch_size=1
steps_per_epoch = len(x_train)/batch_size

history_final_autoencoder = final_autoencoder.fit(x_train, y_train, 
                                                  batch_size=batch_size, 
                                                  steps_per_epoch=steps_per_epoch, 
                                                  epochs=epochs, 
                                                  validation_data=(x_val, y_val),  
                                                  verbose=2, callbacks=[eagerly_stopping])

plt.plot(history_final_autoencoder.history['accuracy'], label='train')
plt.plot(history_final_autoencoder.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.plot(history_final_autoencoder.history['loss'], label='train')
plt.plot(history_final_autoencoder.history['val_loss'], label='validation')
plt.plot('Loss')
plt.legend()
plt.show()

final_autoencoder.save('training_one_autoencoder_at_a_time.h5')

del first_ae, second_ae, final_autoencoder

final_autoencoder = load_model('training_one_autoencoder_at_a_time.h5')

idx = np.random.randint(x_test.shape[0])
test = x_test[idx].reshape((28,28))

prediction = final_autoencoder.predict(test)

plt.imshow(test, cmap='gray')
title = 'Actual:'+labels[np.argmax(y_test[idx])]+' Predicted:'+labels[np.argmax(prediction[0])]
plt.title(title)
plt.show()