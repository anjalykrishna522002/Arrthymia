import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Flatten, LSTM, TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

# Set parameters
num_classes = 6
batch_size = 45
epochs = 100

# Function to read dataset
def read_dataset():
    data_list = []
    label_list = []
    i = 0
    my_list = os.listdir(r'dataset')

    for pa in my_list:
        print(pa, "==================", i)
        for root, dirs, files in os.walk(r'dataset\\' + pa):
            for f in files:
                file_path = os.path.join(r'dataset\\' + pa, f)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
                data_list.append(res)
                label = i
                label_list.append(label)
        i += 1

    return np.asarray(data_list, dtype=np.float32), np.asarray(label_list)

# Load dataset
x_dataset, y_dataset = read_dataset()
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.5, random_state=0)

# Convert labels to categorical
y_train = np.array([to_categorical(i, num_classes) for i in y_train])
y_test = np.array([to_categorical(i, num_classes) for i in y_test])

# Normalize and reshape inputs
x_train = X_train.astype('float32') / 255
x_test = X_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

# Define CNN + LSTM model
def create_cnn_lstm_model():
    model = Sequential()

    # CNN Layers
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Flatten before LSTM
    model.add(TimeDistributed(Flatten()))

    # LSTM Layer
    model.add(LSTM(64, return_sequences=False))

    # Fully connected layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Train CNN + LSTM Model
model_cnn_lstm = create_cnn_lstm_model()
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

model_cnn_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not os.path.exists("cnnlstmmodel1.h5"):
    model_cnn_lstm.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
    model_cnn_lstm.save("cnnlstmmodel1.h5")
else:
    model_cnn_lstm.load_weights("cnnlstmmodel1.h5")

# Evaluate CNN + LSTM Model
y_pred_cnn_lstm = np.argmax(model_cnn_lstm.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy_cnn_lstm = np.mean(y_pred_cnn_lstm == y_true)
print("CNN + LSTM Model Accuracy:", accuracy_cnn_lstm)
