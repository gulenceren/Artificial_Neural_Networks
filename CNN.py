import numpy as np
import os
from skimage import io
import numpy
import pickle
from keras.utils.np_utils import to_categorical
import cv2
import tensorflow.keras as tf
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers import Flatten
import keras
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils as u
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
dataYol ="./Images"
kategoriler=["cicek","meyve","ucak"]
img_size=70
train=[] ; test=[] ; 
def create_traning_data():
    for kategori in kategoriler:
        path=os.path.join(dataYol,kategori) #datasetin icerisindeki kategorilerin klasör yollari
        sinif_sayisi=kategoriler.index(kategori)
        sayac=0
        imgsay= len(os.listdir(path))
        imgsay=imgsay * 0.8
        for img in os.listdir(path):
            try:    
                sayac = sayac + 1
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)            
                yeni_dizi = cv2.resize(img_array,(img_size,img_size))
                if imgsay>sayac:
                    train.append([yeni_dizi,sinif_sayisi])
                else:
                    test.append([yeni_dizi,sinif_sayisi])
            except Exception as e:
                pass
create_traning_data()
print((len(train)))
import random
random.shuffle(train)
random.shuffle(test)
X =[];y=[];test_X=[];test_y=[]
for features, label in train:
    X.append(features)
    y.append(label)
for features, label in test:
    test_X.append(features)
    test_y.append(label)
X = np.array(X).reshape(-1,img_size,img_size,1)
test_X=np.array(test_X).reshape(-1,img_size,img_size,1)

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
pickle_out=open("test_X.pickle","wb")
pickle.dump(test_X,pickle_out)
pickle_out.close()
pickle_out=open("test_y.pickle","wb")
pickle.dump(test_y,pickle_out)
pickle_out.close()

X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))
test_X=pickle.load(open("test_X.pickle","rb"))
test_y=pickle.load(open("test_y.pickle","rb"))
X, test_X = X.astype('float32')/255.0, test_X.astype('float32')/255.0
y = to_categorical(y,3)
test_y = to_categorical(test_y,3)
y=numpy.array(y)
print (y.shape)

model = Sequential() 
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], padding='same',
                 activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
acc=0.1
def predict(model1):
    Pred=model1.predict(test_X)
    Pred = np.around(Pred)
    score, acc = model1.evaluate(test_X, test_y, batch_size=20)
    f1=f1_score(test_y, Pred,average='weighted')
    print(classification_report(test_y, Pred)) 
    print("ACC=",acc)      
    return acc,f1
for i in range(3):
    if (i==0):
        print("=========================ADAM===========================")
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if (i==1):
        print("=========================SGD===========================")
        opt = optimizers.SGD(lr=0.01, clipnorm=1.)
    if (i==2):
        print("=========================ADAMAX===========================")
        opt=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
  
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(test_X, test_y, epochs=5,batch_size=20)
    mAcc,fs=predict(model)
    if mAcc > acc:
        acc = mAcc
        model.save("Model.model")
        if i==0:
            eb=fs
            ebacc=acc
            tutk=i
            model.save("Model.model")
        if (fs>=eb):
            if (fs==eb):
                if (acc>ebacc):
                    ebacc=acc
                    tutk=i
                    model.save("Model.model")
            else:
                eb=fs#ortmse
                tutk=i
                model.save("Model.model")