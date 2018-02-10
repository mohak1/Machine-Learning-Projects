import numpy as np

#sigmoid function
def sigmoid(n):
    return 1/(1+np.exp(-n))

#derivative of sigmoid function
def deriv(n): 
   return n*(1-n)

#input data
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#output data
Y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)
#Declare weights or synapses
w1 = 2*np.random.random([3,4])-1
w2 = 2*np.random.random([4,1])-1

#training code
for i in range(60000):

    l0 = X
    l1 = sigmoid(np.dot(l0,w1))
    l2 = sigmoid(np.dot(l1,w2))

    l2_Error = Y-l2
    l1_Error = Y-l1


    #Back Propogation
    l2_Delta = l2_Error*deriv(l2)
    l1_Error = l2_Delta.dot(w2.T)
    l1_Delta = l1_Error*deriv(l1)

    #Gradient Descent (update parameters)
    w2 += l1.T.dot(l2_Delta)
    w1 += l0.T.dot(l1_Delta)
    
print("Output after training")
print(str(l2))
