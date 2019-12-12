from liegroups.numpy import SE3

import numpy as np

T = np.array([[0, 0, -1, 0.1],
                  [0, 1, 0, 0.5],
                  [1, 0, 0, -0.5],
                  [0, 0, 0, 1]])



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
xi = np.vstack((xi_0, xi_1, xi_2, xi_3, xi_4, xi_5))

#w_bar = np.empty([N,6])

w_bar_0 = np.array([1.0,1.0,1.0,0.0,0.0,0.0]) 
w_bar_1 = np.array([2.0,2.0,2.0,0.0,0.0,0.0]) 
w_bar_2 = np.array([1.0,1.0,1.0,0.0,0.0,0.0]) 
w_bar_3 = np.array([1.0,3.0,1.0,0.0,0.0,0.0]) 
w_bar_4 = np.array([1.0,1.0,4.0,0.0,0.0,0.0])
w_bar_5 = np.array([1.0,1.0,2.0,0.0,0.0,0.0])

w_bar = np.vstack((w_bar_0, w_bar_1, w_bar_2, w_bar_3, w_bar_4, w_bar_5))


# define a dataset of time values
t = np.empty([N,1])

t[0,0] = 0.0
t[1,0] = 1.0
t[2,0] = 2.0
t[3,0] = 3.0
t[4,0] = 4.0
t[5,0] = 5.0

# xi are lists of 6 real-valued entries

def w_bar(v,omega):
    return np.array([v[0],v[1],v[2], omega[0],omega[1],omega[2]])

def error_i(xi_i, xi_ip1,w_bar_i,w_bar_ip1):

    T_i = SE3.exp(xi_i)
    T_ip1 = SE3.exp(xi_ip1)

    e_i1 = SE3.vee(SE3.log(T_ip1.dot(T_i.inv())))
            - (t_ip1-t_i)*w_bar_i

    e_i2 = SE3.inv_left_jacobian(SE3.vee(SE3.log(T_ip1.dot(T_i.inv()))))*w_bar_ip1 - w_bar_i

    return np.vstack = [e_i1, e_i2]


def error_ij():
    # define later in case of measurement-based model

def Q_i(t_i,t_im1,dim):

    delta_t_i = t_i - t_im1

    return np.array([(1/3)*((delta_t_i)**3)*Q_C(dim), (1/2)*((delta_t_i)**2)*Q_C(dim);
                     (1/2)*((delta_t_i)**2)*Q_C(dim), (delta_t_i)*Q_C(dim)])



# TODO
"""
    DONE
    setup xi and t data containers
    defined the cost function structure
    come up with matrix Q_C
    
    #TODO
    do some research on the gauss-newton algorithm
    define dataset xi above
    define w_bar dataset above (lin and rotational velocity)

"""

def Q_C(dim):
    return np.eye(dim)



def J(xi, w_bar, N, Q_C, t):

    # create error vector
        # fill up error vector with all training data
    e = np.empty([N,1])

    for i in range(N):
        e.append(error_i(xi[i], xi[i+1],w_bar[i],w_bar[i+1]))

    # create Q matrix and invert it
    
    # create a vector of all Q_i entries
    Q_vect = np.empty([N,1])

    for i in range(N):
        Q_vect.append(Q_i(Q_C(6),t[i],t[i-1]))
    
    Q = np.diag(Q_vect)

    Q_inv = np.linalg.inv(Q)

    return (1/2)*np.matmul(np.matmul(np.tranpose(e),Q_inv),e)

#system_to_solve 



# implementation of E:

def F(k):
    
    F = np.empty([24,12])

    dim = 6

    T_kp1 = SE3.exp(xi[k+1])
    T_k = SE3.exp(xi[k])

    tau_bar_kp1_k = (SE3.exp([xi[k+1]]).dot(SE3.exp([xi[k]]))).adjoint()

    # lie jacobian is 6x6
    # xi is 6x1
    # T is element 4x4
    # Ad(T)=curly T is element 6x6


    F_k00 = SE3.inv_left_jacobian(T_kp1.dot(T_k)).dot(tau_bar_kp1_k)
    F_k10 = (1/2)*w_bar(v_kp1,omega_kp1).dot(SE3.inv_left_jacobian(T_kp1.dot(T_k))).dot(tau_bar_kp1_k):
    F_k01 = (t_kp1-tk)*np.eye(dim) 
    F_k11 = np.eye(dim)
    F_k02 = -SE3.inv_left_jacobian(T_kp1.dot(T_k))
    F_k12 = (-1/2)*SE3.vee(w_bar(v_kp1,omega_kp1)).dot(SE3.inv_left_jacobian(T_kp1.dot(T_k)))
    F_k03 = np.zeros(dim)
    F_k13 = -SE3.inv_left_jacobian(T_kp1.dot(T_k))

    F_0 = np.block([F_k00, F_k01, F_k02, F_k03])
    F_1 = np.block([F_k10, F_k11, F_k12, F_k13])



    F = np.block(([F_0],[F_1]))

    return F


def A_pri(idx):
    dim = 6
    np.transpose(F(t[idx])).dot(np.linalg.inv(Q_i(idx,idx-1,dim)).dot(F(t[idx])))

def b_pri(idx):
    np.transpose(F(t[idx])).dot(np.linalg.inv(Q_i(idx,idx-1,dim)).dot(error_i(xi[idx,:], xi[idx+1,:],w_bar[idx,:],w_bar[idx+1,:])))





"""

perturbation: epsilon*

initial pose: identity transform


per iteration we update: x_op <- x_op + delta_x_op


T_i = 

"""
xi_init = np.array([0,0,0,0,0,0])
T_init = T_i = SE3.exp(xi_init)

v_init = np.array([0.5, 0.5, 0.5])
omega_init = np.array([0.0, 0.0, 0.0])
w_bar_init = w_bar(v_init, omega_init)

e_op_init = error_i(xi_init, xi[0], w_bar_init, w_bar[0])


error_i(xi_i, xi_ip1,w_bar_i,w_bar_ip1):

def delta_i(theta_i,psi_i):
    return np.array([theta_i, psi_i]) 

def epsilon():
    delta = np.empty([N,1])
    for i in range(N):
        delta.append(delta_i(theta[i], psi[i]))
    
    return epsilon



from pyslam.problem import Problem, Options

options = Options()
options.print_summary = True

problem = Problem(options)






def residuals(self, b):
    """Evaluate the residuals f(x, b) - y with the given parameters.
    Parameters
    ----------
    b : tuple, list or ndarray
        Values for the model parameters.
    Return
    ------
    out : ndarray
        Residual vector for the given model parameters.
    """
    x, y = self.xvals, self.yvals
    return self._numexpr(x, *b) - y





def jacobian(self, b):
    
    """Evaluate the model's Jacobian matrix with the given parameters.
    Parameters
    ----------
    b : tuple, list or ndarray
        Values for the model parameters.
    Return
    ------
    out : ndarray
        Evaluation of the model's Jacobian matrix in column-major order wrt
        the model parameters.
    """
    
    # Substitute parameters in partial derivatives
    subs = [pd.subs(zip(self._b, b)) for pd in self._pderivs]
    # Evaluate substituted partial derivatives for all x-values
    vals = [sp.lambdify(self._x, sub, "numpy")(self.xvals) for sub in subs]
    # Arrange values in column-major order
    return np.column_stack(vals)




def gauss_newton(sys, x0, tol = 1e-10, maxits = 256):
    
    """Gauss-Newton algorithm for solving nonlinear least squares problems.
    Parameters
    ----------
    sys : Dataset
        Class providing residuals() and jacobian() functions. The former should
        evaluate the residuals of a nonlinear system for a given set of
        parameters. The latter should evaluate the Jacobian matrix of said
        system for the same parameters.
    x0 : tuple, list or ndarray
        Initial guesses or starting estimates for the system.
    tol : float
        Tolerance threshold. The problem is considered solved when this value
        becomes smaller than the magnitude of the correction vector.
        Defaults to 1e-10.
    maxits : int
        Maximum number of iterations of the algorithm to perform.
        Defaults to 256.
    Return
    ------
    sol : ndarray
        Resultant values.
    its : int
        Number of iterations performed.
    Note
    ----
    Uses numpy.linalg.pinv() in place of similar functions from scipy, both
    because it was found to be faster and to eliminate the extra dependency.
    """


    dx = np.ones(len(x0))   # Correction vector
    xn = np.array(x0)       # Approximation of solution

    i = 0
    while (i < maxits) and (dx[dx > tol].size > 0):
        # correction = pinv(jacobian) . residual vector
        dx  = np.dot(np.linalg.pinv(sys.jacobian(xn)), -sys.residuals(xn))
        xn += dx            # x_{n + 1} = x_n + dx_n
        i  += 1

    return xn, i