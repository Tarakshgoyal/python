import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization,Dense,InputLayer
from tensorflow.keras.losses import MeanSquaredError,Huber,MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
data=pd.read_csv("C:/Users/Taraksh Goyal/Desktop/coding/python/tensorflow/train.csv",sep=",")
print(data.head())
#sns.pairplot(data[['v.id','on road old','on road now','years','km','rating','condition','economy','top speed','hp','torque','current price']],diag_kind='kde')
#plt.show()
tensor_data=tf.constant(data)
tensor_data=tf.cast(tensor_data,tf.float32)
tensor_data=tf.random.shuffle(tensor_data)
#print(tensor_data[:5])
X=tensor_data[:,3:-1]
y=tensor_data[:,-1]
y=tf.expand_dims(y,axis=-1)
#print(y[:5])
normalizer = Normalization()
normalizer.adapt(X)
X_normalized = normalizer(X)
#print(X_normalized[:5])
model=tf.keras.Sequential([InputLayer(input_shape=(8,)),normalizer,Dense(1),])
print(model.summary())
model.summary()
tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True)
model.compile(optimizer=Adam(learning_rate=0.1),loss=MeanAbsoluteError(),metrics=[RootMeanSquaredError()])
TRAIN_RAIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
DATASET_SIZE=len(X)
train_size = int(DATASET_SIZE * TRAIN_RAIO)
val_size = int(DATASET_SIZE * VAL_RATIO)
X_train = X_normalized[:train_size]
y_train = y[:train_size]
x_val = X_normalized[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
x_test = X_normalized[train_size + val_size:]
y_test = y[train_size + val_size:]
model.evaluate(x_test,y_test)
model.fit(X_train,y_train,validation_data=(x_val,y_val),epochs=100,verbose=1)
model.predict(x_test)
y_pred=list(model.predict(x_test)[:,0])
ind=np.arange(100)
plt.figure(figsize=(40,12))
width=0.4
y_test_np = y_test.numpy().flatten()
plt.bar(ind,y_pred,width,label='Predicted Car Price')
plt.bar(ind+width,y_test_np,width,label='Actual car price')
plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('Car price Prices')
plt.show()