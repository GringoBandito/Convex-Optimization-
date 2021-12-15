import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp

mean1 = np.ones((25,1))*-2
mean2 = np.ones((25,1))*-1
mean3 = np.ones((25,1))*1
mean4 = np.ones((25,1))*2

mean = np.concatenate((mean1,mean2))
mean = np.concatenate((mean,mean3))
mean = np.concatenate((mean,mean4))
X = np.random.normal(mean,1,(100,100))

beta = np.random.normal(1,1,(100,1))
epsilon = np.random.normal(0,1,(100,1))


y = np.matmul(X,beta) + epsilon


def dykstra(X,y,lambd,epochs):
    
    length,width = np.shape(X)
    u = np.zeros((length,width-1))
    u = np.append(u,y,axis=1)
    count = 0
    z = np.zeros((length,width))
    lst = []
    while count < epochs:
        
        
        for i in range(width):
            uu = cp.Variable((length))
            yy = u[:,i-1] + z[:,i]
            obj = cp.sum_squares(yy-uu)
            constr = [np.transpose(X[:,i])@uu <= lambd]
            
            prob = cp.Problem(cp.Minimize(obj),constr)
            
            prob.solve()
            
            u[:,i] = uu.value
            
            z[:,i] = u[:,i-1] + z[:,i] - u[:,i]
            
        
        
        count += 1
    
    
    
    
    return u,y,z
            
            
    
#res1,res2,res3 = dykstra(X,y,.01,15)

def block_descent_dykstra(X,beta,y,iterations):
    
    length,width = np.shape(X)
    weight = np.zeros((width,1))
    u = np.zeros((length,width-1))
    u = np.append(u,y,axis=1)
    z = np.zeros((length,width))
    resvec = np.zeros((length,1))
    count = 0
    lambd = .1
    yaxis = []
    yaxis2 = []
    xaxis = []
    while count < iterations:
        
        for i in range(len(weight)):
            
            uu = cp.Variable((length))
            yy = u[:,i-1] + z[:,i]
            obj = cp.sum_squares(yy-uu)
            constr = [np.transpose(X[:,i])@uu <= lambd]
            
            prob = cp.Problem(cp.Minimize(obj),constr)
            
            prob.solve()
            
            u[:,i] = uu.value
            
            z[:,i] = u[:,i-1] + z[:,i] - u[:,i]
            
            minvalue = 10000000000000000000000000000000000
            block = np.linspace(-5,5,num=400)
            np.append(block,0)
            for j in block:
                value = y - np.matmul(X[:,0:i],weight[0:i]) - np.matmul(X[:,i+1:],weight[i+1:]) - (X[:,i] * j)
                value2 = .5 * (np.linalg.norm(value) ** 2) + lambd*abs(j)
                
                if value2 <= minvalue:
                    minvalue = value2
                    weight[i] = j 
            
            
        spread = u - y + np.matmul(X,weight)
        yaxis.append(np.linalg.norm(spread)/(length*width))
        
        
        for i in range(width):
            spread2 = z[:,i] - X[:,i]*weight[i]
            resvec[i] = np.linalg.norm(spread2)/len(spread2)
        
        yaxis2.append(np.linalg.norm(resvec)/len(resvec))
        
            
        
        xaxis.append(count)
        count += 1
    
   
    
    return u,yaxis,yaxis2,xaxis

betavector,dual,residual,epochnumb = block_descent_dykstra(X,beta,y,40)

plt.plot(epochnumb,dual,color='red')
plt.plot(epochnumb,residual,color = 'blue')

plt.xlabel('Iteration')
plt.ylabel('Mse')
plt.title('Dykstra vs BCD N=100, s=p=100')
plt.legend(['Dual','Residual Vector'])
plt.savefig('Dykstra vs BCD3')
plt.show()
