from keras.datasets  import mnist
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np


(train_feature, train_lable), (test_feature , test_label) = mnist.load_data()
print (len(train_feature),len(train_lable))
print (train_feature.shape,train_lable.shape)

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()

show_image(train_feature[3])
print(train_lable[3])

###feature preprocess###################

train_feature_vector = train_feature.reshape(len(train_feature),784).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')
print (train_feature_vector.shape,test_feature_vector.shape)

####normalize#############

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

####label preprocess #########

print (train_lable[0:5])



train_lable_onehot = to_categorical(train_lable)
test_lable_onehot = to_categorical(test_label)
print (train_lable_onehot)

#### create sequential model ####

from keras.models import Sequential
from keras.layers import Dense


model = Sequential()

### create input layer and hidden layer#########

model.add(Dense (units= 256, input_dim=784,kernel_initializer='normal',activation= 'relu'))

model.add(Dropout(0.2))

###second hidden laryer ######

model.add(Dense(units=128,kernel_initializer='normail',activation='relu'))

model.add(Dropout(0.2))

### create output layer ###############

model.add(Dense (units = 10 , kernel_initializer='normal', activation= 'softmax'))

### training model #####################

model.compile (loss= 'categorical_crossentropy', optimizer= 'adam',metrics=['accuracy'])

train_history = model.fit(x=train_feature_normalize,y=train_lable_onehot,validation_split=0.2,epochs=10,batch_size=200,verbose=2) 

### accuracy check #####

scores = model.evaluate(test_feature_normalize,test_lable_onehot)

print ('\準確率 = ',scores[1])

prediction = np.argmax(model.predict(test_feature_normalize),axis=1)


show_image(test_feature[0])
print (test_label[0],prediction[0])

model.save('Mnist_mlp_model.h5')