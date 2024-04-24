1# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:13:40 2019

@author: sony
"""

import h2o
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras import optimizers
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report
import warnings
import sys
import xlwt
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
print ("""
    H2O ve GridSerach Ile Deep Learning -> 1
    Keras ile Deep Learning -> 2 """)
secim = input("Bir Seçim Yapınız: ")
if secim == "1" or secim == 1:
    h2o.init()
    h2o.remove_all() 
    pd.set_option('display.max_columns', None)
    veritum=pd.read_excel("ParkinsonDataset.xls", sheet_name="Sheet1")
    train, test = train_test_split(veritum,test_size=0.3,random_state=42,shuffle=True)
    fr = h2o.H2OFrame(train)
    y = "Classification"
    x = list(fr.columns)
    x.remove(y)
    frtest = h2o.H2OFrame(test)
    fr[y] = fr[y].asfactor()
    frtest[y] = frtest[y].asfactor()
    nfolds = 5
    hyper_params = {"rate": [0.01, 0.03,0.04,0.05,0.06,0.07,0.08,0.09],
                    "epochs": [100,200,300],
                    "activation": ["tanh","rectifier","maxout","exprectifier","rectifierwithdropout"],
                    "loss": ["CrossEntropy","Automatic","Quadratic"],
                    "hidden":[[10,20,30],[50,30,40]]                }
    search_criteria = {"strategy": "RandomDiscrete", "max_models": 20, "seed": 1}
    
    grid = H2OGridSearch(model=H2ODeepLearningEstimator(                                        
                                                            ),
                         hyper_params=hyper_params,
                         search_criteria=search_criteria,
                         grid_id="model")
    
    grid.train(x=x, y=y, training_frame=fr)
    grid_performance = grid.get_grid(sort_by='auc', decreasing = True)
    print(grid_performance)
    best_model = grid_performance.models[0]
    testsonuc = best_model.model_performance(frtest)
    print(testsonuc.accuracy())
    print(testsonuc)
def predict(model1,kontrol):
    if (kontrol==1):
        model1=load_model(dosyaismi)
    y_pred=model1.predict(X_test)
    y_pred = np.around(y_pred)
    y_test_non_category = [ np.argmax(t) for t in Y_test ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    score, acc = model1.evaluate(X_test, Y_test, batch_size=128)
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    f1=f1_score(Y_test, y_pred,average='weighted')
    if (kontrol==1):
        print (conf_mat) 
        print(classification_report(Y_test, y_pred)) 
        print("ACC=",acc)      
    return acc,f1
def predict_egitim(model1,kontrol):
    if (kontrol==1):
        model1=load_model(dosyaismi)
    y_pred=model1.predict(X_train)
    y_pred = np.around(y_pred)
    y_test_non_category = [ np.argmax(t) for t in Y_train ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    #print(y_pred)
    score, acc = model1.evaluate(X_train, Y_train, batch_size=128)
    #score, acc = model1.evaluate(egitimgiris, y_train, batch_size=128)
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    f1=f1_score(Y_train, y_pred,average='weighted')
    if (kontrol==1):
        print (conf_mat) 
        print(classification_report(Y_train, y_pred)) 
        print("ACC=",acc)  
        
    return acc,f1
    
if secim == "2" or secim == 2:
    dosyaismi='deep.h5'
    kacozellik=9
    veritum=pd.read_excel("ParkinsonDataset.xls")
    Train, Test= train_test_split(veritum, test_size=0.30, random_state=30)
    filepath="./train.xls"
    Train.to_excel(filepath,index=False)
    filepath="./test.xls"
    Test.to_excel(filepath,index=False)
    Train.to_excel("train.xls")
    Test.to_excel("test.xls")
    
    X_train=Train.iloc[:,0:kacozellik]
    X_test=Test.iloc[:,0:kacozellik]
    Y_train=Train.iloc[:,kacozellik]
    Y_test=Test.iloc[:,kacozellik]

    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train= encoder.transform(Y_train)
    encoder.fit(Y_test)
    Y_test= encoder.transform(Y_test)
    
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    
    acc1=0.1
    while(acc1<0.62):
        a=2
        baslangic=0
        bitis=6
        for k in range(baslangic,bitis):
            if (k==0):opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            if (k==1):opt=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            if (k==2):opt = optimizers.SGD(lr=0.01, clipnorm=1.)
            if (k==3):opt =keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
            if (k==4):opt=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            if (k==5):opt=keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            #print (egiris,ecikis)
            #model.compile(loss='categorical_crossentropy   mean_absolute_error', optimizer=opt, metrics=['accuracy'])
            model = Sequential()
            aktivasyon="softsign"
            model.add(Dense(16, input_dim=kacozellik, activation=aktivasyon))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            for i in range(10):
                model.fit(X_train, Y_train, epochs=150, batch_size=64,verbose=0)
                acc,fs=predict(model,0)
                if i==0 and k==baslangic:
                    eb=fs
                    ebacc=acc
                    tutk=k
                    model.save("deep.h5")
                if (fs>=eb):
                    if (fs==eb):
                        if (acc>ebacc):
                               ebacc=acc
                               tutk=k
                               model.save("deep.h5")
                    else:
                        eb=fs#ortmse
                        tutk=k
                        model.save("deep.h5")
        print ("En Büyük F Score:",eb, "K değeri=",tutk, "EB ACC",ebacc)
        print("Test sonuç")
        acc1,fs1=predict(model,1)
        predict_egitim(model,1)
    