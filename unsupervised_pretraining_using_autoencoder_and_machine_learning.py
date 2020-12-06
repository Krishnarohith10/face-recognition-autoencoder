import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.python.keras.api.keras import optimizers, losses
from tensorflow.python.keras.api.keras.callbacks import EarlyStopping
from tensorflow.python.keras.api.keras.models import Sequential, load_model
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
le = LabelEncoder()
train_outputs = le.fit_transform(train_outputs)

shuffle_index = np.random.permutation(len(train_inputs))
train_inputs, train_outputs = train_inputs[shuffle_index], train_outputs[shuffle_index]

x_train, x_test, y_train, y_test = train_test_split(train_inputs, train_outputs, test_size=0.2)

del train_inputs, train_outputs

epochs = 200
batch_size=32

autoencoder = Sequential()
autoencoder.add(Input(shape=x_train.shape[1:]))
autoencoder.add(Dense(50, activation='relu'))
autoencoder.add(Dropout(0.4))
autoencoder.add(Dense(10, activation='relu'))
autoencoder.add(Dropout(0.4))
autoencoder.add(Dense(x_train.shape[-1], activation='relu'))

early_stopping = EarlyStopping(patience=5, monitor='val_loss', mode='min', restore_best_weights=True)

autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
history_autoencoder = autoencoder.fit(x_train, x_train,
                                      callbacks=[early_stopping], 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_split=0.1,
                                      verbose=1)

plt.plot(history_autoencoder.history['loss'],  label='train')
plt.plot(history_autoencoder.history['val_loss'], label='validation')
plt.legend()
plt.show()

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(x_train, y_train)

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)

print('Confusion Matrix:', confusion_matrix(y_train, y_train_pred))

plt.matshow(confusion_matrix(y_train, y_train_pred), cmap=plt.cm.gray)
plt.show()

print('Precision Score:', precision_score(y_train, y_train_pred, average='micro'), 
      '\nRecall Score:', recall_score(y_train, y_train_pred, average='micro'), 
      '\nF1 Score', f1_score(y_train, y_train_pred, average='micro'))
