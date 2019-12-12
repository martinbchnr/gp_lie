from liegroups.numpy import SE3

import numpy as np

T = np.array([[0, 0, -1, 0.1],
                  [0, 1, 0, 0.5],
                  [1, 0, 0, -0.5],
                  [0, 0, 0, 1]])



# number of data points for training
N = 4

# define a dataset of xi_vectors with points rho and rotations phi
xi = np.empty([N,6])

w_bar = np.empty([N,6])

# define a dataset of time values
t = np.empty([N,1])
 

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

def Q_i(Q_C,t_i,t_im1):

    delta_t_i = t_i - t_im1

    return np.array([(1/3)*((delta_t_i)**3)*Q_C, (1/2)*((delta_t_i)**2)*Q_C;
                     (1/2)*((delta_t_i)**2)*Q_C, (delta_t_i)*Q_C])



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