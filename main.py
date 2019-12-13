from liegroups.numpy import SE3

import numpy as np

"""
T = np.array([[0, 0, -1, 0.1],
                  [0, 1, 0, 0.5],
                  [1, 0, 0, -0.5],
                  [0, 0, 0, 1]])
"""



# number of data points for training
N = 6

# define a dataset of xi_vectors with points rho and rotations phi
#xi = np.empty([N,6])
xi_0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
xi_1 = np.array([1.0,1.0,0.0,0.0,0.0,0.0])
xi_2 = np.array([2.0,2.0,0.0,0.0,0.0,0.0])
xi_3 = np.array([3.0,3.0,0.0,0.0,0.0,0.0])
xi_4 = np.array([4.0,4.0,0.0,0.0,0.0,0.0])
xi_5 = np.array([5.0,5.0,0.0,0.0,0.0,0.0])
xi = [xi_0, xi_1, xi_2, xi_3, xi_4, xi_5]

#w_bar = np.empty([N,6])

w_bar_0 = np.array([1.0,1.0,1.0,0.0,0.0,0.0]) 
w_bar_1 = np.array([2.0,2.0,2.0,0.0,0.0,0.0]) 
w_bar_2 = np.array([1.0,1.0,1.0,0.0,0.0,0.0]) 
w_bar_3 = np.array([1.0,3.0,1.0,0.0,0.0,0.0]) 
w_bar_4 = np.array([1.0,1.0,4.0,0.0,0.0,0.0])
w_bar_5 = np.array([1.0,1.0,2.0,0.0,0.0,0.0])

w_bar = [w_bar_0, w_bar_1, w_bar_2, w_bar_3, w_bar_4, w_bar_5]

w_bar_est_init = np.array([1.0,1.0,1.0,0.0,0.0,0.0]) 
w_bar_est = w_bar_est_init

w_bar_mean = np.array([1.0,0.0,0.0,0.0,0.0,0.0]) 

T = [SE3.exp(xi_0),SE3.exp(xi_1),SE3.exp(xi_2),SE3.exp(xi_3),SE3.exp(xi_4),SE3.exp(xi_5)]

#T_test = SE3.exp(xi_1) 
#test = SE3.log(T)

# define a dataset of time values
t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# xi are lists of 6 real-valued entries

haha1 = SE3.curlywedge(w_bar_mean)

def Q_C(dim):
    return np.identity(dim)


def Phi(w_bar,t,t_km1):
    
    #eq10.22
    
    matrix00 = np.identity(6) + (t-t_km1) * SE3.curlywedge(w_bar_mean)
    matrix01 = (t-t_km1) * np.identity(6) + (1/2)*((t-t_km1)**(2)) * SE3.curlywedge(w_bar_mean)
    matrix10 = np.zeros((6,6))
    matrix11 = np.identity(6)

    matrix0 = np.block([matrix00, matrix01])
    print(matrix0.shape)
    matrix1 = np.block([matrix10, matrix11])
    print(matrix1.shape)
    
    matrix = np.concatenate((matrix0,matrix1))
    print(str(matrix.shape) + "dim of phi")
    return matrix #np 12x12 array

    



def Q_k(k,t,w_bar_mean,dim):
    
    # equation 10.24a
    
    #return np.array([(1/3)*((delta_t_i)**3)*Q_C(dim), (1/2)*((delta_t_i)**2)*Q_C(dim),
    #                (1/2)*((delta_t_i)**2)*Q_C(dim), (delta_t_i)*Q_C(dim)])

    delta_t = t[k] - t[k-1]
    
    #w_bark = w_bar[k]

    Q_k11 = (1/3)*(delta_t**(3))*Q_C(dim) \
            + (1/8)*(delta_t**(4))*(np.dot(SE3.curlywedge(w_bar_mean),Q_C(dim)) + np.dot(Q_C(dim),np.transpose(SE3.curlywedge(w_bar_mean)))) \
             + (1/20)*(delta_t**(5))*np.dot(SE3.curlywedge(w_bar_mean), np.dot(Q_C(dim),np.transpose(SE3.curlywedge(w_bar_mean))))
    
    Q_k12 = (1/2)*(delta_t**(2))*Q_C(dim) + (1/6)*(delta_t**(3))*np.dot(SE3.curlywedge(w_bar_mean),Q_C(dim))
    Q_k22 = (delta_t**(1))*Q_C(dim)
    
    row0 = np.block([Q_k11, Q_k12])
    print("Q_k: row0 shape")
    print(row0.shape)
    
    print("Q_k: row1 shape")
    row1 = np.block([np.transpose(Q_k12), Q_k22])
    print(row1.shape)
    
    Q_k_matrix = np.concatenate((row0,row1))
    print("Q_k: Q_k shape")
    print(Q_k_matrix.shape)
    
    return Q_k_matrix
    
    

def F_inv(K,w_bar):
    #eq10.17
    F_inv = np.identity((K+1)*12)
    for k in range(K):
        F_inv[(k+1)*12:12+(k+1)*12 , k*12:12+k*12] = -Phi(w_bar[k],k+1,k)
        
    return F_inv # returns ((K+1)*12,(K+1)*12)-np array


T_op = SE3.exp(xi_0)
w_bar_op = w_bar_0
T_est = [T_op,T_op,T_op,T_op,T_op,T_op]


def e_v(T_est,w_bar, w_bar_est,K):
    
    # vee turns SE3 back into se3
    
    #eq 10.7

    e_v0_0 = SE3.log(SE3.exp(xi[0]).dot(T_est[0].inv()))
    e_v0_1 = w_bar[0] - w_bar_est
    
    e_v0 = -np.concatenate((e_v0_0, e_v0_1))
    print(str(e_v0.shape) + "dim of e_v0")
    
    e_v = e_v0
    
    for k in range(1,K+1):
        
        T_k = SE3.exp(xi[k])
        T_km1 = SE3.exp(xi[k-1])
        
        bracket1_0 = SE3.log(T_km1.dot((T_est[k-1]).inv()))
        bracket1_1 = w_bar[k-1] - w_bar_est
        #print (bracket1_0.shape)
        #print (bracket1_1.shape)
        
        bracket1 = np.concatenate((bracket1_0,bracket1_1))
        #print (bracket1.shape)
        
        bracket2_0 = SE3.log(T[k].dot((T_est[k]).inv()))
        bracket2_1 = w_bar[k] - w_bar_est
        #print (bracket2_0.shape)
        #print (bracket2_1.shape)
        
        bracket2 = np.concatenate((bracket2_0,bracket2_1))
        #print (bracket1.shape)
        
        e_vk = np.dot(Phi(w_bar[k],t[k],t[k-1]),bracket1) - bracket2
        
        print(str(e_vk.shape) + "dim of e_vk")
        print(str(e_v.shape) + "dim of e_v")
        
        
        e_v = np.concatenate((e_v,e_vk))
    
    print(str(e_v.shape) +  "dim of e_v")
    
    return e_v


#Q = np.diag(np.diag(np.identity(60)))


def Q(K,t,w_bar_mean,dim):

    matrix = np.empty([K*12+12,K*12+12])
    matrix[0:12,0:12] = np.identity(12)
    for k in range(1,K):
        matrix[k*12:k*12+12, k*12:k*12+12] = Q_k(k,t,w_bar_mean,dim)
        
    print("final Q shape")
    print(matrix.shape)
    return matrix #(K*12+12,K*12+12)-np array

#test_2 = SE3.log(SE3.exp(xi[0]).dot(T_est[0].inv()))


test_e_v = e_v(T_est,w_bar, w_bar_op,5)



# eq 10.35
prod1 = np.transpose(F_inv(5,w_bar))
prod2 = np.linalg.inv(Q(5,t,w_bar_mean, 6))
prod3 = e_v(T_est,w_bar,w_bar_op,5)


prod4 = np.dot(prod2,prod3)

print("dims of prods")
print(prod1.shape)
print(prod2.shape)
print(prod3.shape)
print(prod4.shape)

b = np.dot(prod1,prod4)

A = np.dot(np.transpose(F_inv(5,w_bar)),np.dot(prod2,F_inv(5,w_bar)))

#solve the cost-function equation-system


dx = np.dot(np.linalg.inv(A),b)

