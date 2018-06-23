import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data, converters convert the letter to a number
data= np.loadtxt('dataset/letter-recognition.data', dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})

#Shuffle data randomly
np.random.shuffle(data)

# split the data 80:20
train = data[:int(len(data)*0.80)]
test = data[int(len(data)*0.80):]

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

# Initiate the kNN, classify, measure accuracy.
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/len(test)
print(accuracy)
