import os,cv2,glob
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense

######## data input and got label #######################
images = [] 
labels = []

dict_labels={"Cat":0 , "Dog":1}
size = (40,40)

for folders in glob.glob("./kagglecatsanddogs_5340/PetImages/*"):
    print (folders,"image loading")
    for filename in os.listdir(folders):
        label=folders.split("/")[-1]
        print (label)
        try:
            img=cv2.imread(os.path.join(folders,filename))
            if img is not None:
                img = cv2.resize(img,dsize=size)
                images.append(img)
                labels.append(dict_labels[label])
        except:
            print (os.path.join(folders,filename),"can't load")
            pass

print (len(images),len(labels))

#### train data and test data split ############

train_feature,test_feature,train_label,test_label = train_test_split(images,labels,test_size=0.2,random_state=42)


train_feature = np.array(train_feature)
test_feature = np.array(test_feature)
train_label = np.array(train_label)
test_label = np.array(test_label)

print (len(train_feature),len(test_feature))
print (train_feature.shape,train_label.shape)
print (test_feature.shape,test_label.shape)

##### save data in npy file  ##############

imagesavepath="Cat_Dog_Dataset/"
if not os.path.exists(imagesavepath):
    os.mkdir(imagesavepath)

np.save(imagesavepath+"train_feature.npy",train_feature)
np.save(imagesavepath+"test_feature.npy",test_feature)
np.save(imagesavepath+"train_label.npy",train_label)
np.save(imagesavepath+"test_label.npy",test_label)

#### reshape the data ####################

train_feature_vector = train_feature.reshape(len(train_feature),40,40,3).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature),40,40,3).astype('float32')

#### normalize the data ########

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

#### categorical the labels #######

train_label_onehot = to_categorical(train_label)
test_label_onehot = to_categorical(test_label)

#### Set the model for CNN #####

model = Sequential()

model.add(Conv2D(filters=10,kernel_size=(5,5),padding='same',input_shape=(40,40,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.1))

model.add(Conv2D(filters=20,kernel_size=(5,5),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=512,activation='relu'))

model.add(Dense(units=2,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#### fit the model #############

train_history= model.fit(x=train_feature_normalize,y=train_label_onehot,validation_split=0.2,epochs=20,batch_size=200,verbose=2)

scores = model.evaluate(test_feature_normalize,test_label_onehot)
print ("scores = ", scores[1])

#### prediction with test data ########

prediction = model.predict(test_feature_normalize)
prediction = np.argmax(prediction,axis=1)

print (test_label[0:10],prediction[0:10])

#### save the model ##############

model.save('cat_dog_CNN_model.h5')