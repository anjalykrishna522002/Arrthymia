import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import uuid

def read_dataset1(path):
    data_list = []
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    
    return np.asarray(data_list, dtype=np.float32)

def predict_cnn_lstm(fn):
    K.clear_session()
    
    # Read and preprocess the dataset
    dataset = read_dataset1(fn)
    dataset = dataset.reshape(dataset.shape[0], 48, 48, 1)  # Reshape for CNN input
    dataset /= 255  # Normalize
    
    # Load CNN-LSTM combined model
    cnn_lstm_model = load_model("cnnlstmmodel1.h5")
    cnn_lstm_pred_prob = cnn_lstm_model.predict(dataset)[0]  # Get probability distribution
    cnn_lstm_pred = np.argmax(cnn_lstm_pred_prob)  # Get highest class
    
    print(f"CNN + LSTM Prediction: {cnn_lstm_pred}")
    
    # Generate and save the plot as an image
    plot_image_path = "static/plots/" + str(uuid.uuid4()) + "_plot.png"
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(cnn_lstm_pred_prob)), cnn_lstm_pred_prob, color='blue', alpha=0.6)
    plt.xticks(range(len(cnn_lstm_pred_prob)), [f'Class {i}' for i in range(len(cnn_lstm_pred_prob))])
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("CNN + LSTM Model Prediction Probabilities")
    plt.ylim(0, 1)  # Probabilities range from 0 to 1
    plt.savefig(plot_image_path)  # Save the plot to an image file
    plt.close()  # Close the plot to release memory
    
    K.clear_session()
    
    return cnn_lstm_pred, plot_image_path  