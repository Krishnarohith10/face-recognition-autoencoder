import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.api.keras.models import load_model, Sequential, Model
from tensorflow.python.keras.api.keras.layers import Input, Dropout, Conv2DTranspose, Flatten, Reshape
from tensorflow.python.keras.api.keras.layers import BatchNormalization, Conv2D, Dense, Activation
from tensorflow.python.keras.api.keras import optimizers, losses
from tensorflow.python.keras.api.keras.callbacks import EarlyStopping

path = 'D:\\face_recognition\\dataset'
labels = os.listdir(path)

train_inputs = []
train_outputs = []

for label in labels:
    for image in tqdm(os.listdir(os.path.join(path, label))):
        img = cv2.imread(os.path.join(path, label, image))
        img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        img = img.reshape(-1)
        train_inputs.append(img)
        train_outputs.append(label)

train_inputs = np.array(train_inputs).astype('float32')/255.0
le = LabelEncoder()
train_outputs = le.fit_transform(train_outputs)

shuffle_index = np.random.permutation(len(train_inputs))
train_inputs, train_outputs = train_inputs[shuffle_index], train_outputs[shuffle_index]

x_train, x_test, y_train, y_test = train_test_split(train_inputs, train_outputs, test_size=0.2)

del train_inputs, train_outputs

epochs = 200
batch_size=32

class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            Input(shape=(input_shape)), 
            Conv2D(16, (3,3), (2,2), 'same'), 
            Dropout(0.4), 
            BatchNormalization(), 
            Activation('relu'), 
            Conv2D(16, (3,3), (2,2), 'same'), 
            Dropout(0.4), 
            BatchNormalization(), 
            Activation('relu'), 
            Conv2D(8, (3,3), (2,2), 'same'), 
            Dropout(0.4), 
            BatchNormalization(), 
            Activation('relu'), 
            ])
        self.decoder = Sequential([
            Conv2DTranspose(8, (3,3), (2,2), 'same'), 
            Activation('relu'), 
            Conv2DTranspose(16, (3,3), (2,2), 'same'), 
            Activation('relu'), 
            Conv2DTranspose(16, (3,3), (2,2), 'same'), 
            Activation('relu'), 
            Conv2D(3, (3,3), (1,1), 'same'), 
            Activation('sigmoid')
            ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(x_train.shape[1:])

early_stopping = EarlyStopping(patience=5, monitor='val_loss', mode='min', restore_best_weights=True)

autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
history_autoencoder = autoencoder.fit(x_train, x_train,
                                      callbacks=[early_stopping], 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_split=0.1, verbose=1)

plt.plot(history_autoencoder.history['loss'], label='train')
plt.plot(history_autoencoder.history['val_loss'], label='validation')
plt.legend()
plt.show()
