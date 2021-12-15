import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

X = np.random.normal([0,1,2,2,1,0,0,1,2,0],1,(500,10))

beta = np.array([17,18,-17,-18,17.5,17.3,-18.3,-18.3,-17.7,-17.9])
epsilon = np.random.normal(0,1,(500,1))


y = np.matmul(X,beta) + epsilon



def block_descent(X,beta,y,iterations):
    
    length,width = np.shape(X)
    weight = np.zeros((width,1))
    count = 0
    lambd = .1
    yaxis = []
    xaxis = []
    while count < iterations:
        
        for i in range(len(weight)):
            minvalue = 10000000000000000000000000000000000
            block = np.linspace(-4,4,num=400)
            np.append(block,0)
            for j in block:
                value = y - np.matmul(X[:,0:i],weight[0:i]) - np.matmul(X[:,i+1:],weight[i+1:]) - (X[:,i] * j)
                value2 = .5 * (np.linalg.norm(value) ** 2) + lambd*abs(j)
                
                if value2 <= minvalue:
                    minvalue = value2
                    weight[i] = j 
            
        
        spread = np.linalg.norm(beta - weight)/len(weight)
        yaxis.append(spread)
        xaxis.append(count)
        
        
                
        count += 1
    
    signcount = 0
        
    for i in range(len(weight)):
        if np.sign(beta[i]) == np.sign(weight[i]): 
            signcount += 1
    
    
    return weight,yaxis,xaxis,signcount/len(weight)

#xx,yy,zz,aa = block_descent(X,beta,y,40)

def support_montecarlo(X,beta,y,iterations):
    lst = []
    for i in range(20):
        XX = np.random.normal([0,1,2,2,1,0,0,1,2,0],1,(500,10))

        betaa = np.random.normal(1,1,(10,1))
        epsilonn = np.random.normal(0,1,(500,1))
        yy = np.matmul(XX,betaa) + epsilonn
        
        
        a,b,c,d = block_descent(XX,betaa,yy,iterations)
        lst.append(d) 
        
    return lst

tots = support_montecarlo(X,beta,y,13)
  

'''
plt.plot(zz,yy,color='red')

plt.xlabel('Iteration')
plt.ylabel('Mse')
plt.title('Coordinate Descent Beta L2 Error N=25,s=50,p=100')
plt.savefig('CD_N500,s=50,p=100')
plt.show()
'''
#plt.savefig('CD_N500,s=p=5,10')


'''
plt.hist(tots,bins=10,edgecolor='black')
plt.title('Coordinate Descent Beta Support Recovery N=500,s=5,p=10')
plt.xlabel('Matching Element Percentage')
plt.ylabel('Proportion of Samples')

plt.savefig('histo.n500.p10.s5')
plt.show()
'''