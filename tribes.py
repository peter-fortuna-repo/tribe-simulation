# tribes.py
# Peter Fortuna
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp

def define_ivp_system(n, C):
    '''
    This returns a function defining the differential equation (ode) system
    :param n: ( n, ) np.ndarray difference between birth rate and death rate of the ith tribe
    :param C: ( n, n ) np.ndarray matrix containing migration rate where C[i,j]
                       represents the migration rate from j to i (diagonal is 0)
    :param I: ( n, n ) np.ndarray matrix containing interactions between tribes
    :return: a callable ode system
    '''

    def ode(t, P):
        '''
        :param t: independent variable (i.e. time)
        :param P: population change as a function of time
        :return: (np.ndarray) ode system
        '''
        system = np.zeros_like(n)
        size = system.size
        for i in range(size):
            system[i] = P[i] * (n[i] + sum([(C[i,j]-C[j,i])*P[j] for j in range(size)]))
        return system
    return ode


def numerical_solve(P0, t_span, M, n, C):
    '''
    Numerical solves the ivp using solve_ivp
    :parama P0: (n, ) np.ndarray array containing the initial value
    :param t_span: (tuple) contains the starting and ending time value
    :param disc: ( int ) temporal discretization
    :param n: (n, ) np.ndarray difference between birth rate and death rate of the ith relgion
    :param C: (n, n) np.ndarray matrix containing conversion rate differences
    :param I: (n, n) np.ndarray matrix containing interactions between tribes
    :param method ="ivp": boolean determing the method
    :return: callable the ode system
    '''
    ode = define_ivp_system(n, C)
    t_eval = np.linspace(t_span[0], t_span[-1], M)
    sol =  solve_ivp(ode, t_span, P0, t_eval=t_eval)

    return sol


def plot_sol(P0, t_span, M, n, C, method='ivp', labels=[]):
    '''
    plots the solution
    '''
    sol = numerical_solve(P0, t_span, M, n, C, method=method)
    fig = plt.figure()
    fig.set_dpi(300)
    ax = fig.add_subplot(111)
    for s, lab in zip(sol.y, labels):
        ax.plot(sol.t, s, label=lab)
        ax.legend(loc='best')

    ax.set_xlabel('Years')
    ax.set_ylabel('Population')
    ax.set_title('SIR Model of tribes')
    plt.show()
    return None