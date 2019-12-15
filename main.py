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

w_bar_init = np.array([1.0,1.0,1.0,0.0,0.0,0.0]) 
w_bar = [w_bar_0, w_bar_1, w_bar_2, w_bar_3, w_bar_4, w_bar_5]

w_bar_est = w_bar_init

#w_bar_mean = np.array([1.0,1.0,0.0,0.0,0.0,0.0]) 

T = [SE3.exp(xi_1),SE3.exp(xi_1),SE3.exp(xi_1),SE3.exp(xi_1),SE3.exp(xi_1),SE3.exp(xi_1)]


#T_test = SE3.exp(xi_1) 
#test = SE3.log(T)

# define a dataset of time values
t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# xi are lists of 6 real-valued entries


def Q_C(dim):
    Q_C = np.identity(dim)
    Q_C[0,0] = 0.1
    Q_C[1,1] = 0.01
    Q_C[2,2] = 0.1
    Q_C[3,3] = 0.001
    Q_C[4,4] = 0.001
    Q_C[5,5] = 0.01
    return np.identity(dim)

def Q_k_new_inv(t,k,dim):
    
    delta_t = t[k] - t[k-1]
    
    # equations 10.25
    Q_k11 = 12*((delta_t)**(-3))*np.linalg.inv(Q_C(dim))
    Q_k12 = -6*((delta_t)**(-2))*np.linalg.inv(Q_C(dim))
    Q_k22 = 4*((delta_t)**(-1))*np.linalg.inv(Q_C(dim))
    
    row0 = np.block([Q_k11, Q_k12])
    row1 = np.block([Q_k12, Q_k22])
    
    Q_k_matrix = np.concatenate((row0,row1))
    return Q_k_matrix

def Q_new_inv(K,t,dim):

    matrix = np.empty([K*12,K*12])

    for k in range(0,K):
        matrix[k*12:k*12+12, k*12:k*12+12] = Q_k_new_inv(t,k,dim)
        
    print("final Q shape")
    print(matrix.shape)
    return matrix #(K*12+12,K*12+12)-np array


def Phi(w_bar,t,t_km1):
    
    #eq10.22
    
    matrix00 = np.identity(6) + (t-t_km1) * SE3.curlywedge(w_bar_est)
    matrix01 = (t-t_km1) * np.identity(6) + (1/2)*((t-t_km1)**(2)) * SE3.curlywedge(w_bar_est)
    matrix10 = np.zeros((6,6))
    matrix11 = np.identity(6)

    matrix0 = np.block([matrix00, matrix01])
    print(matrix0.shape)
    matrix1 = np.block([matrix10, matrix11])
    print(matrix1.shape)
    
    matrix = np.concatenate((matrix0,matrix1))
    print(str(matrix.shape) + "dim of phi")
    return matrix #np 12x12 array

    



def Q_k(k,t,w_bar_est,dim):
    
    # equation 10.24a
    
    delta_t = t[k] - t[k-1]
    
    
    # equations 10.25
    Q_k11 = (1/3)*(delta_t**(3))*Q_C(dim) \
            + (1/8)*(delta_t**(4))*(np.dot(SE3.curlywedge(w_bar_est),Q_C(dim)) + np.dot(Q_C(dim),np.transpose(SE3.curlywedge(w_bar_est)))) \
             + (1/20)*(delta_t**(5))*np.dot(SE3.curlywedge(w_bar_est), np.dot(Q_C(dim),np.transpose(SE3.curlywedge(w_bar_est))))
    
    Q_k12 = (1/2)*(delta_t**(2))*Q_C(dim) + (1/6)*(delta_t**(3))*np.dot(SE3.curlywedge(w_bar_est),Q_C(dim))
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
T_est = T


def e_v(T_est,w_bar, w_bar_est,K,theta):
    
    # vee turns SE3 back into se3
    
    #eq 10.7

    e_v0_0 = -SE3.log(SE3.exp(xi[0]).dot(T_est[0].inv()))
    e_v0_1 = -(w_bar[0] - w_bar_est)
    
    e_v0 = np.concatenate((e_v0_0, e_v0_1))
    print(str(e_v0.shape) + "dim of e_v0")
    
    e_v = e_v0 - theta[0:12]
    
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
        
        
        e_vk = e_vk + np.dot(Phi(w_bar[k],t[k],t[k-1]), theta[(k-1)*12:(k-1)*12+12]) - theta[k*12:k*12+12]
        
        print(str(e_vk.shape) + "dim of e_vk")
        #print(str(e_v.shape) + "dim of e_v")
        
        
        e_v = np.concatenate((e_v,e_vk))
    
    print(str(e_v.shape) +  "dim of e_v")
    
    return e_v

def e_v_new(T_est,w_bar, K):
    
    # vee turns SE3 back into se3
    
    #eq 10.7


    for k in range(0,K):
        
        a1 = SE3.log(T_est[k+1].dot(T_est[k].inv())) - (t[k+1]-t[k])*w_bar[k]
        a2 = np.dot(np.linalg.inv(SE3.left_jacobian(SE3.log(T_est[k+1].dot(T_est[k].inv())))),w_bar[k+1])-w_bar[k]
        
        a = np.concatenate((a1,a2))
        if k==0:
            e_v = np.concatenate((a1,a2))
        else:
            e_v = np.concatenate((e_v,a))
    
    print(str(e_v.shape) +  "dim of e_v")
    
    return e_v


#Q = np.diag(np.diag(np.identity(60)))


def Q(K,t,w_bar_est,dim):

    matrix = np.empty([K*12+12,K*12+12])
    matrix[0:12,0:12] = np.identity(12)
    for k in range(0,K):
        matrix[k*12:k*12+12, k*12:k*12+12] = Q_k(k,t,w_bar_est,dim)
        
    print("final Q shape")
    print(matrix.shape)
    return matrix #(K*12+12,K*12+12)-np array

#test_2 = SE3.log(SE3.exp(xi[0]).dot(T_est[0].inv()))


### MEASUREMENT MODEL

# landmarks

# G_1
    
# G_2
    
# R





dx= np.empty([72])


# eq 10.35
prod1 = np.transpose(F_inv(5,w_bar))
prod2 = np.linalg.inv(Q(5,t,w_bar_est, 6))
prod3 = e_v(T_est,w_bar,w_bar_op,5,dx)


prod4 = np.dot(prod2,prod3)

print("dims of prods")
print(prod1.shape)
print(prod2.shape)
print(prod3.shape)
print(prod4.shape)

b = np.dot(prod1,prod4)

A = np.dot(np.transpose(F_inv(5,w_bar)),np.dot(prod2,F_inv(5,w_bar)))

def E(T_est, w_bar, K,dim):
      

    for k in range(0,K):
        E_k00 = np.dot(np.linalg.inv(SE3.left_jacobian(SE3.log(T_est[k+1].dot(T_est[k].inv())))), SE3.adjoint(T_est[k+1].dot(T_est[k].inv())))
        E_k01 = -(t[k+1]-t[k])*np.identity(dim)
        E_k02 = np.linalg.inv(SE3.left_jacobian(SE3.log(T_est[k+1].dot(T_est[k].inv()))))
        E_k03 = np.zeros((dim,dim))
        E_k10 = (-1/2)*np.dot(SE3.curlywedge(w_bar[k+1]), np.dot(np.linalg.inv(SE3.left_jacobian(SE3.log(T_est[k+1].dot(T_est[k].inv())))), \
                                                         SE3.adjoint(T_est[k+1].dot(T_est[k].inv()))))
        E_k11 = -np.identity(dim)
        E_k12 = (1/2)*np.dot(SE3.curlywedge(w_bar[k+1]), np.linalg.inv(SE3.left_jacobian(SE3.log(T_est[k+1].dot(T_est[k].inv())))))
        E_k13 = np.linalg.inv(SE3.left_jacobian(SE3.log(T_est[k+1].dot(T_est[k].inv()))))
        
        
        
        E_k0 = np.block([E_k00,E_k01, E_k02, E_k03])
        
        E_k1 = np.block([E_k10,E_k11, E_k12, E_k13])
        E_k = np.concatenate((E_k0,E_k1))
        
        if k==0:
            E = E_k
        else:
            E = np.concatenate((E,E_k))
    
    print("E final shape")
    print(E.shape)
    return E

    

#solve the cost-function equation-system


maxits = 100
i = 0
K = 5

T_outcome = []
    
while (i < maxits):
    
    '''
    # calculate A and b
    prod1 = np.transpose(F_inv(5,w_bar))
    prod2 = np.linalg.inv(Q(5,t,w_bar_est, 6))
    prod3 = e_v(T_est,w_bar,w_bar_est,5,dx)
    prod4 = np.dot(prod2,prod3)
    
    print("dims of prods")
    print(prod1.shape)
    print(prod2.shape)
    print(prod3.shape)
    print(prod4.shape)
    
    b = np.dot(prod1,prod4)
    A = np.dot(np.transpose(F_inv(5,w_bar)),np.dot(prod2,F_inv(5,w_bar)))
    
    # calculate correction through matrix inversion dx=A(^-1)b
    dx = np.dot(np.linalg.inv(A),b)
    '''
    
    c1 = np.transpose(E(T_est, w_bar, K,6))
    c2 = Q_new_inv(K,t,6)
    c3 = E(T_est, w_bar, K,6)
    
    print("shapes of components of A:")
    print(c1.shape)
    print(c2.shape)
    print(c3.shape)
    
    d1 = np.transpose(E(T_est, w_bar, K,6))
    d2 = Q_new_inv(K,t,6)
    d3 = e_v_new(T_est,w_bar,K)
    
    print("shapes of components of b:")
    print(d1.shape)
    print(d2.shape)
    print(d3.shape)
    
    A = np.dot(np.transpose(E(T_est, w_bar, K,6)),np.dot(Q_new_inv(K,t,6),E(T_est, w_bar, K,6)))
    
    
    
    b = -np.dot(np.transpose(E(T_est, w_bar, K,6)), np.dot(Q_new_inv(K,t,6),e_v_new(T_est,w_bar,K)))
    
    dx = np.dot(np.linalg.pinv(A),b)
    
    for k in range(0,K+1):
        
        """
        T_start_idx = k*6
        T_end_idx = k*6+6
        print("--- T_start_index")
        print(T_start_idx)
        print(T_end_idx)
        """
        
        #T_est[k] = SE3.exp(dx[k*6:k*6+6]).dot(T_est[k])
        #w_bar[k] = w_bar[k] + dx[(K+1)*6+k*6:(K+1)*6+k*6+6]
        
        T_est[k] = SE3.exp(dx[k*12:k*12+6]).dot(T_est[k])
        w_bar[k] = w_bar[k] + dx[k*12+6:k*12+12]
        
        """
        w_start_idx = (K+1)*6+k*6
        w_end_idx = (K+1)*6+k*6+6
        
        print("--- T_end_index")
        print(w_start_idx)
        print(w_end_idx)
        """
        
        if k==1:
            T_outcome.append(SE3.log(T_est[k]))
        
        
    i = i + 1
    
        
# transform the estimated SE3 groups back to Euclidean space
points = []
for entry in T_est:
    points.append(SE3.log(entry))


### Querying the trajectory







