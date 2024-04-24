from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas
import numpy as np
import seaborn as sns

df=pandas.read_excel(open('./ParkinsonDataset.xls','rb'))   
array=df.values
array2=[]
for i,m in enumerate(array):
    array2.append(m)
array2=np.array(array2)
np.random.shuffle(array2)       
df=pandas.DataFrame(array2)
filepath="./shuffleParkinson.xls"
df.to_excel(filepath,index=False)
        
X=array2[:,0:len(df.columns)-1]
y=array2[:,len(df.columns)-1]
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.25, random_state=27)
print("======================lbfgs==================== ")
clf = MLPClassifier(activation='relu',hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='lbfgs', verbose=10,  random_state=21,tol=0.000000001)
               #max_iter=epochs degeridir
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("Acc: ",acc)
print("============Confusion Matrix=========== ")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ",cm)
confusion_matrix = pandas.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins = True)
sns.heatmap(confusion_matrix, annot=True)
