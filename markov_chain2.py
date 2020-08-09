import numpy as np
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
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
from math import radians, cos, sin, asin, sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import h5py




num_class = 4
num_classes = num_class


def resort(raw):
    processed = [];
    for i in range(0,len(raw)):
        if raw[i] == 0:
            processed.append(0)
        elif raw[i] == 1:
            processed.append(1)
        else:
            processed.append(0)

    return processed



##########################################################################
data = np.array(pd.read_excel('new_vulture_data.xlsx'))
data = pd.DataFrame(data)
data = data.fillna(method='ffill')
data = np.array(data)

#disney, mac,sarkis,morongo,rosalie
rosalie = data[data[:,13]=='Rosalie']

rosalie_label = rosalie[:,11]
rosalie_input  = rosalie[:,2:4]







rosalie= rosalie[:,16:]
rosalie_input  =  np.hstack((rosalie_input, rosalie))


no = 0

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
    
migration_labels = rosalie_label
rosalie_label_gd = []
for j in range(0,len(dist)):
    if dist[j] < 1:
        rosalie_label_gd.append(0)
    else:
        rosalie_label_gd.append(1)


#make sure you compare for different thresholds of move and don't move 0.6km, 0.8km,1km,1.2km,1.5km,2km
rosalie_label = []
for j in range(0,len(dist)):
    if dist[j] < 1 and migration_labels[j] == 2:
        rosalie_label.append(0)
        
    elif dist[j] < 1 and migration_labels[j] == 4:
        rosalie_label.append(0)
    
    elif dist[j] < 1 and migration_labels[j] == 1:
        rosalie_label.append(2)
    
    elif dist[j] < 1 and migration_labels[j] == 3:
        rosalie_label.append(3)
        
    else:
        rosalie_label.append(1)
        




rosalie_label = np.array(rosalie_label[no:len(rosalie_label)])
rosalie_label_gd = np.array(rosalie_label_gd[no:len(rosalie_label_gd)])
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
print('unique',np.unique(ytrain))


def listToString(s):
    str1 = " "

    return (str1.join(s))



class MarkovChain(object):
   def __init__(self, transition_matrix, states):
       """
       Initialize the MarkovChain instance.

       Parameters
       ----------
       transition_matrix: 2-D array
           A 2-D array representing the probabilities of change of
           state in the Markov Chain.

       states: 1-D array
           An array representing the states of the Markov Chain. It
           needs to be in the same order as transition_matrix.
       """
       self.transition_matrix = np.atleast_2d(transition_matrix)
       self.states = states
       self.index_dict = {self.states[index]: index for index in
                          range(len(self.states))}
       self.state_dict = {index: self.states[index] for index in
                          range(len(self.states))}

   def next_state(self, current_state):
       """
       Returns the state of the random variable at the next time
       instance.

       Parameters
       ----------
       current_state: str
           The current state of the system.
       """
       return np.random.choice(
        self.states,
        p=self.transition_matrix[self.index_dict[current_state], :]
       )

   def generate_states(self, current_state, no=10):
       """
       Generates the next states of the system.

       Parameters
       ----------
       current_state: str
           The state of the current random variable.

       no: int
           The number of future states to generate.
       """
       future_states = []
       for i in range(no):
           next_state = self.next_state(current_state)
           future_states.append(next_state)
           current_state = next_state
       return future_states
    





transitions = list(map(str,ytrain))



def rank(c):
    return ord(c) - ord('0')

T = [rank(c) for c in transitions]

#create matrix of zeros

M = [[0]*4 for _ in range(4)]

for (i,j) in zip(T,T[1:]):
    M[i][j] += 1

#now convert to probabilities:
for row in M:
    n = sum(row)
    if n > 0:
        row[:] = [f/sum(row) for f in row]


g = 0
for row in M:
    if g == 0:
        acc = row;g = g+1;
    else:
        acc = np.vstack((acc,row))
        
print(acc)
transition_matrix = acc

weather_chain = MarkovChain(transition_matrix=transition_matrix,states=['0', '1', '2', '3'])


no = 18

ytest_ = list(map(str,ytest))
#result = [ytest_[0]];
result = []
for i in range(0, len(ytest_)-no):
    acc = []
    a = ytest_[i]
    for j in range(0,no):
        a = weather_chain.next_state(current_state= a);
    
    result.append(a)




test_list = [int(i) for i in result]

ytest = ytest[no:len(ytest)]
test_list = resort(test_list);

acc = np.abs(test_list - rosalie_label_gd[le+no:len(rosalie_label_gd):])
acd = 1-(np.count_nonzero(acc)/len(ytest))
print("accuracy is ", acd)
precision, recall, fscore, support = score(test_list,rosalie_label_gd[le+no:len(rosalie_label_gd):])
print('fscore: {}'.format(fscore))



print('time in advance (dual,first instance)', no)
