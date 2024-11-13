import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,InputLayer,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError
def splits(dataset, train_ratio, val_ratio, test_ratio):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)    
    train_dataset = dataset.take(train_size)
    val_test_dataset = dataset.skip(train_size)
    val_dataset = val_test_dataset.take(val_size)
    test_dataset = val_test_dataset.skip(val_size)   
    return train_dataset, val_dataset, test_dataset
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_dataset, val_dataset, test_dataset = splits(dataset[0], train_ratio, val_ratio, test_ratio)
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_dataset.take(16)):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
plt.show()
IM_SIZE=224
def resizing(image,label):
    return tf.image.resize(image,(IM_SIZE,IM_SIZE))/255,label
train_dataset=train_dataset.map(resizing)
val_dataset=val_dataset.map(resizing)
test_dataset=test_dataset.map(resizing)
for image,label in train_dataset.take(1):
    print(image,label)
train_dataset=train_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset=val_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset=test_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
model=tf.keras.Sequential([InputLayer(input_shape=(IM_SIZE,IM_SIZE,3)),Conv2D(filters=6,kernel_size=3,strides=1,padding='valid',activation='relu'),BatchNormalization(),MaxPool2D(pool_size=2,strides=2),Conv2D(filters=16,kernel_size=3,strides=1,padding='valid',activation='relu'),BatchNormalization(),MaxPool2D(pool_size=2,strides=2),Flatten(),Dense(1000,activation="relu"),BatchNormalization(),Dense(100,activation="relu"),BatchNormalization(),Dense(1,activation="sigmoid"),])
print(model.summary())
y_true=[0,1,0,0]
y_pred=[0.6,0.51,0.94,1]
y_true = tf.convert_to_tensor(y_true)
y_pred = tf.convert_to_tensor(y_pred)
bce=tf.keras.losses.BinaryCrossentropy()
bce(y_true,y_pred)
model.compile(optimizer=Adam(learning_rate=0.01),loss=BinaryCrossentropy(),metrics=['accuracy'])
history=model.fit(train_dataset,validation_data=val_dataset,epochs=100,verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy','val_accuracy'])
plt.show()
test_dataset=test_dataset.batch(1)
model.evaluate(test_dataset)
def parasite_or_not(x):
    if(x<0.5):
        return str('P')
    else:
        return str('U')
p=0
unin=1
parasite_or_not(model.predict(test_dataset.take(1))[0][0])
for i,(image,label) in enumerate(test_dataset.take(9)):
    ax=plt.subplot(3,3,i+1)
    plt.imshow(image[0])
    plt.title(str(parasite_or_not(label.numpy()[0])) + ":" + str(parasite_or_not(model.predict(image)[0][0])))
    plt.axis('off')