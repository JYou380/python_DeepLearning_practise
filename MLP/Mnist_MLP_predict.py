import glob,cv2
import numpy as np
from keras.models import load_model


files = glob.glob("imagedata\*.jpg")

test_feature=[]
test_label=[]

for file in files:
    img=cv2.imread(file)
    img=cv2.cvtColor(ing,cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    test_feature.append(img)
    label=file[10:11]
    test_label.append(int(label))

test_feature = np.array(test_feature)
test_label = np.array(test_label)

test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')

test_feature_normalize = test_feature_vector/255

print ("loading model")
model = load_model('mnist_mlp_model.h5')

#predict

prediction = model.predict_classes(test_feature_normalize)

