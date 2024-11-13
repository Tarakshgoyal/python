import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import datetime
import pathlib
import io
import os
import time
import random
#from google.colab import files
from PIL import Image
import albumentations as A
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
train_directory="C:/Users/Taraksh Goyal/Desktop/coding/python/tensorflow/Emotions Dataset/Emotions Dataset/train"
validation_directory="C:/Users/Taraksh Goyal/Desktop/coding/python/tensorflow/Emotions Dataset/Emotions Dataset/test"
CLASS_NAMES=["angry","happy","sad"]
CONFIGURATION={
    "BATCH_SIZE" : 32,
    "IM_SIZE" : 256,
    "LEARNING_RATE" : 0.001,
    "N_EPOCHS" : 20,
    "DROPOUT_RATE" : 0.0,
    "REGULARIZATION_RATE" : 0.0,
    "N_FILTERS" : 6,
    "KERNEL_SIZE" : 3,
    "N_STRIDES" : 1,
    "POOL_SIZE" : 2,
    "N_DENSE_1" : 100,
    "N_DENSE_2" : 10,
    "NUM_CLASSES" : 3,
    }
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=99,
)
val_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    validation_directory,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)
for i in val_dataset.take(1):
    print(i)
plt.figure(figsize=(12,12))
for images,labels in train_dataset.take(1):
    for i in range(16):
        ax=plt.subplot(4,4,i+1)
        plt.imshow(images[i]/255.)
        plt.title(CLASS_NAMES[tf.argmax(labels[i],axis=0).numpy()])
        plt.axis("off")
plt.show()
training_dataset=(train_dataset.prefetch(tf.data.AUTOTUNE))
validation_dataset=(val_dataset.prefetch(tf.data.AUTOTUNE))
resize_rescale_layers=tf.keras.Sequential([Resizing(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"]),Rescaling(1./255),])
lenet_model=tf.keras.Sequential([
    InputLayer(input_shape=(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"],3)),
    Rescaling(1./255,name="rescaling"),
    Conv2D(filters=CONFIGURATION["N_FILTERS"],kernel_size=CONFIGURATION["KERNEL_SIZE"],strides=CONFIGURATION["N_STRIDES"],
        activation='relu',kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D(pool_size=CONFIGURATION["POOL_SIZE"],strides=CONFIGURATION["N_STRIDES"]*2),
    Dropout(rate=CONFIGURATION["DROPOUT_RATE"]),
    Conv2D(filters=CONFIGURATION["N_FILTERS"]*2 +4,kernel_size=CONFIGURATION["KERNEL_SIZE"],strides=CONFIGURATION["N_STRIDES"],
    activation='relu',kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),BatchNormalization(),
    MaxPool2D(pool_size=CONFIGURATION["POOL_SIZE"],strides=CONFIGURATION["N_STRIDES"]*2),
    Flatten(),
    Dense(CONFIGURATION["N_DENSE_1"],activation="relu",kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(rate=CONFIGURATION["DROPOUT_RATE"]),
    Dense(CONFIGURATION['N_DENSE_2'],activation='relu',kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dense(CONFIGURATION["NUM_CLASSES"],activation='softmax'),
])
print(lenet_model.summary())
loss_function=CategoricalCrossentropy(from_logits=False)
y_true=[[0,1,0],[0,0,1]]
y_pred=[[0.05,0.95,0],[0.1,0.8,0.1]]
y_true = tf.convert_to_tensor(y_true)
y_pred = tf.convert_to_tensor(y_pred)
cce=tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true,y_pred).numpy())
loss_function=CategoricalCrossentropy()
metrics=[CategoricalAccuracy(name="accuracy"),TopKCategoricalAccuracy(k=2,name="top_k_accuracy")]
lenet_model.compile(optimizer=Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),loss=loss_function,metrics=metrics)
history=lenet_model.fit(training_dataset,validation_data=validation_dataset,epochs=CONFIGURATION["N_EPOCHS"],verbose=1,)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()