import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import scipy as sp
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# Display pure signal.
def displayPlot(samples, sample_rate):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('amplitude')
    ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)
    plt.show()


# Create spectrogram from input signal.
def spectrogram_pure(samples, sample_rate, stride_ms=10.0, window_ms=20.0, max_freq=None, eps=1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
    
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram


# Preprocessing data for training. Getting soundwaves and its labels of the given classes.
def preprocessing(labels):
    all_wave = list()
    all_label = list()
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
            if(len(samples) == 16000):
                all_wave.append(samples)
                all_label.append(label)

    return (all_wave, all_label)


# Getting spectograms as numpy arrays in correct dimensions.
def featureMap(all_wave, sample_rate=16000):
    spectrograms = list()
    for wave in all_wave:
        sg = spectrogram_pure(wave, sample_rate, max_freq=16000)
        sg = np.array(sg).reshape(161, 99, 1)
        spectrograms.append(sg)
    return np.array(spectrograms)


# Fetting the correct label formats (categorical).
def lablePreprocessing(all_label, labels):
    # Convert the labels to integer encoded.
    le = LabelEncoder()
    y = le.fit_transform(all_label)
    classes = list(le.classes_)
    
    # Convert labels to one hot vector.
    y = np_utils.to_categorical(y, num_classes=len(labels))
    return (y, classes)


# Builindg the model.
def buildModel(input_data, num_classes, input_shape=(161, 99, 1)):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        normalization_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    
    return model


# Plot losses throuhg training.
def plotLoss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    

# Plot accuracy through training.
def plotAccurecy(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    train_audio_path = 'audio/audio/'
    labels = labels=['up', 'down', 'left', 'right']
    all_wave, all_label = preprocessing(labels)

    spectrograms = featureMap(all_wave)
    y, classes = lablePreprocessing(all_label, labels)
    
    x_train, x_test, y_train, y_test = train_test_split(spectrograms, np.array(y), stratify=y,test_size=0.2, random_state=777, shuffle=True)

    # Model building.
    model = buildModel(x_train, 4)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(x_train, y_train, epochs=10, callbacks=[mc], batch_size=32, validation_data=(x_test, y_test))
    plotLoss(history)
    plotAccurecy(history)