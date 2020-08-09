import keras
from keras.layers import Input, LSTM, Dense, GRU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
import numpy.matlib
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import pandas as pd
from keras.layers import Dense, Activation
import math
from keras.layers import TimeDistributed
import xlrd
import xlsxwriter
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
from math import radians, cos, sin, asin, sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import h5py



def AB_train(Xtrain,ytrain):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(Xtrain, ytrain)
    return clf

def AB_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1-(np.count_nonzero(acc)/len(ytest))
    return acd, ba
    #return ba
    
def RF_train(Xtrain,ytrain):
    #clf = RandomForestClassifier(max_depth=2, random_state=0,class_weight="balanced")
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(Xtrain, ytrain)
    return clf
    
def RF_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1-(np.count_nonzero(acc)/len(ytest))
    return acd, ba
    #return ba
 

def LR_train(Xtrain,ytrain):
    #clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',class_weight="balanced").fit(Xtrain, ytrain)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(Xtrain, ytrain)
    return clf
    
def LR_test(clf, Xtest,ytest):
    ba  = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    print("i am looking ")
    print(acc)
    print(acc.shape)
    acd = 1-(np.count_nonzero(acc)/len(ytest))
    return acd, ba
    #return ba
    
def SV_train(Xtrain,ytrain):
    clf = SVC(gamma='auto')
    clf = clf.fit(Xtrain, ytrain)
    return clf

def SV_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    acc = np.abs(ba - ytest)
    acd = 1-(np.count_nonzero(acc)/len(ytest))
    return acd, ba
    #return ba
def DNN_train(Xtrain,ytrain):
    num_class = 2
    num_features = 28
    n_epoch = 20
    n_batch = 100

    model = Sequential()
    model.add(Dense(256,input_shape=(num_features,)))
    model.add(Dense(256))
    model.add(Dense(num_class))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))

    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
    filepath="weights-improvement3-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    checkpoint = EarlyStopping(monitor='loss', mode='min', verbose=1, patience = 200)
    callbacks_list = [checkpoint,checkpoint1]
    
    ytrain = np.array(pd.get_dummies(np.array(ytrain.astype(int).reshape(-1))))
    model.fit(Xtrain,ytrain, epochs=n_epoch, batch_size=n_batch, verbose=2, callbacks = callbacks_list)
    model.save('monednn'+str(no))
    clf = model
    return clf
    
def DNN_test(clf, Xtest,ytest):
    ba = clf.predict(Xtest)
    print(ba.shape)
    ba = np.argmax(ba, axis=1);
    print(ba)
    print(ytest)
    acc = (ba-ytest);
    acd = 1- (np.count_nonzero(acc)/ len(ytest) )
 
    return acd, ba



look_back = 100
num_features = 28
num_class = 2
num_classes = num_class



##########################################################################
data = np.array(pd.read_excel('new_vulture_data.xlsx'))
data = pd.DataFrame(data)
data = data.fillna(method='ffill')
data = np.array(data)

#disney, mac,sarkis,morongo,rosalie
rosalie = data[data[:,13]=='Rosalie']

rosalie_label = rosalie[:,11]
rosalie_input  = rosalie[:,2:4]




rosalie_label = np.array(rosalie_label)
workbook = xlsxwriter.Workbook('rosalie_label.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(rosalie_label.reshape(len(rosalie_label),1).T):
    worksheet.write_column(row, col, data)
workbook.close()


rosalie= rosalie[:,16:]
rosalie_input  =  np.hstack((rosalie_input, rosalie))


no = 18

print('time in advance (dual,first instance)', no)



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



dist = []
for j in range(0, len(rosalie_input)-1):

    lon1 = rosalie_input[j,0];lon2 = rosalie_input[j+1,0];
    lat1 = rosalie_input[j,1];lat2 = rosalie_input[j+1,1];
    
    dist.append( haversine(lon1, lat1, lon2, lat2) )
    
    
dist = [1] + dist
    
dist = np.array(dist)
workbook = xlsxwriter.Workbook('rosalie_dist.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(dist.reshape(len(dist),1).T):
    worksheet.write_column(row, col, data)
workbook.close()





#make sure you compare for different thresholds of move and don't move 0.6km, 0.8km,1km,1.2km,1.5km,2km

rosalie_label = []
for j in range(0,len(dist)):
    if dist[j] < 1:
        rosalie_label.append(0)
    else:
        rosalie_label.append(1)




rosalie_label = np.array(rosalie_label[no:len(rosalie_label)])
rosalie_input = rosalie_input[0:len(rosalie_input)-no]
print('label', len(rosalie_label))
print('input', len(rosalie_input))




Xtrain = rosalie_input
ytrain = rosalie_label

le = int(0.75*len(ytrain))
print('me')
print(le)
Xtest = Xtrain[le:len(ytrain),:]
ytest = ytrain[le:len(ytrain):]
print('ytest', ytest)
ytest = ytest.astype(int)


Xtrain = Xtrain[0:le,:]
ytrain = ytrain[0:le:]
ytrain_ = ytrain
ytrain = ytrain.astype(int)



"""
clf =  RF_train(Xtrain,ytrain)
acc,acc_ =  RF_test(clf, Xtest, ytest )
print('random forest',acc)
precision, recall, fscore, support = score(ytest, acc_)
print('fscore: {}'.format(fscore))

clf = SV_train(Xtrain,ytrain)
acc,acc_ = SV_test(clf, Xtest, ytest )
print('support vector',acc)
precision, recall, fscore, support = score(ytest, acc_)
print('fscore: {}'.format(fscore))


clf = LR_train(Xtrain,ytrain)
acc,acc_ = LR_test(clf, Xtest, ytest )
print('logistic regression',acc)
precision, recall, fscore, support = score(ytest, acc_)
print('fscore: {}'.format(fscore))
"""




clf = DNN_train(Xtrain,ytrain)
acc,acc_ = DNN_test(clf, Xtest,ytest)
print('DNN',acc)
precision, recall, fscore, support = score(ytest, acc_)
print('fscore: {}'.format(fscore))






