#House Price predictor
import numpy as np
import math

#cost function (J)
def J(X,theta,b,Y):
    H = np.dot(X,theta) + b
    J = (1/20) * np.sum(H-Y)**2
    return J

def Gradient(alpha,b,theta,X):
    H = np.dot(X,theta)
    b -= alpha*(1/10)*np.sum(H-Y.T)
    tem = np.sum(np.dot((H-Y.T),X))
    print("valus of tem "+str(tem))
    theta[0] -= alpha*(1/10)*tem
    theta[1] -= alpha*(1/10)*tem
    theta[2] -= alpha*(1/10)*tem


#input data
X = np.array([[3,1,1],
             [3,2.5,2],
             [2,1,1],
             [4,3,1],
             [3,2,1],
             [3,2.5,2],
             [3,1.5,1],
             [3,1,1],
             [3,2.5,2],
             [3,2.5,2]])

#output data
Y = np.array([[221900],
             [538000],
             [180000],
             [604000],
             [510000],
             [257500],
             [291850],
             [229500],
             [323000],
             [662500]])

np.random.seed(1)

#3 features - 3 parameters
theta = np.random.random([3])
b = 1
alpha = 0.01


#loop
for i in range(0,40000):
    Gradient(alpha,b,theta,X)
    if(i%10000):
        print(str())

inp =np.array([3,1,1])
print(np.dot(theta,inp))
    
