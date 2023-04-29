#### Import Libraries
from ortools.linear_solver import pywraplp
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman"
})
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
viridis = mpl.colormaps['viridis'].resampled(3000)

#### Functions
def solvelp(C_A,C_L,P,X,U,D):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    infinity = solver.infinity()
    pi = {}
    for i in range(X):
        for u in range(U): 
            pi[i,u] =  solver.NumVar(0,infinity, 'pi[%i][%i]'%(i,u))
    constraint = solver.RowConstraint(0, D, '')
    for i in range(X): 
        for u in range(U): 
            constraint.SetCoefficient(pi[i,u],C_L[i][u])
    objective = solver.Objective()

    for i in range(X): 
        for u in range(U):  
            objective.SetCoefficient(pi[i,u], C_A[i][u])

    prob_constraint = [pi[i,u] for i in range(X) for u in range(U)]
    solver.Add(sum(prob_constraint) == 1)
    for j in range(X):
        transition_constraint_left = [pi[j,u] for u in range(U)]
        transition_constraint_right = [P[u][i][j]*pi[i,u] for i in range(X) for u in range(U)]
        solver.Add(sum(transition_constraint_left) == sum(transition_constraint_right))
    objective.SetMinimization()
    status = solver.Solve()
    probpolicy = np.zeros((X,U))
    if status == pywraplp.Solver.OPTIMAL:
        #print('Objective value =', solver.Objective().Value())
        for i in range(X):
            for u in range(U):
                probpolicy[i,u] =  pi[i,u].solution_value()
               #print(pi[i,u].name(), ' = ', pi[i,u].solution_value())
    else:
        print('The problem does not have an optimal solution.')
    return probpolicy
    
def policyfrom(probpolicy,X):
    policy = (probpolicy/probpolicy.sum(axis = 1).reshape(X,1))[:,0]
    return policy
    
def expected(P,C,X,policy,T=10000,A = 2,X_0 = 0):
    state = X_0
    totalcost = 0
    for t in range(T):
        action = np.random.choice([0,1],p=[policy[state],1- policy[state]])
        totalcost += C[state][action]
        state = np.random.choice(np.arange(X),p = P[action][state])
    return totalcost/T
    
 def expectedlength(P,X,policy,L,T=10000,A = 2,X_0 = 0):
    state = X_0
    totalcost = 0
    for t in range(T):
        action = np.random.choice([0,1],p=[policy[state],1- policy[state]])
        totalcost += state%(L)
        #int(state%L)
        state = np.random.choice(np.arange(X),p = P[action][state])
    return totalcost/T
    probpolicy = solvelp(C_A,C_L,P,X,U,D)


def averagecost_randompolicy(cost,n_iter,P,policy,X_0 = 0):
    X = X_0
    A = P.shape[0]
    O = P.shape[2]
    averagecost = np.zeros(n_iter)
    for i in range(n_iter):
      u = policy[X]
      q_k = u[0]
      a_i = int(np.random.choice([0,1],p=[1-q_k,q_k]))
      averagecost[i] = cost[X][a_i]
      X = np.random.choice(np.arange(O),p = P[a_i,X])
    return averagecost.mean()
  
def sigmoid(x,thres=0,scale=1,tau=1):
    return scale/(1 + np.exp(-(x-thres)/tau))
    
def policy_from_sigmoid2d(parameters,L,O,A,tau):
    policy = np.zeros((O*L,1),dtype = float)
    n_thresholds = int(parameters.shape[0]-1)//O
    assert n_thresholds==2
    q = np.sin(parameters[-1])**2
    a=0
    for o in range(O):        
        paras = parameters[o*n_thresholds:(o+1)*n_thresholds]
        thresholds = np.array(paras).reshape(n_thresholds)
        for l in range(L):
            policyvalue = sigmoid(l,thresholds[0],q,tau) + sigmoid(l,thresholds[1],1-q,tau) 
            policy[o*L+l] = policyvalue
    return policy
# SPSA algorithm 
def spsa(initial_parameters,delta,n_iter,T,P,D,lamb,epsilon,rho,L,O,A,C_A,C_L,tau=0.3):
	m = initial_parameters.shape[0]
	parameters = initial_parameters.copy().reshape(m)
	parameters_store = np.zeros((n_iter,m))
	for i in range(n_iter):
	np.random.seed(i)
	pertub = np.random.binomial(1,0.5,(m))
	parameters_plus = parameters + pertub*delta[i]
	parameters_minus = parameters - pertub*delta[i]
	assert parameters_plus.shape == parameters.shape
	policy = policy_from_sigmoid2d(parameters,L,O,A,tau)
	policy_plus = policy_from_sigmoid2d(parameters_plus,L,O,A,tau)
	policy_minus = policy_from_sigmoid2d(parameters_minus,L,O,A,tau)
	C_A_plus = averagecost_randompolicy(C_A,T,P,policy_plus)
	C_A_minus = averagecost_randompolicy(C_A,T,P,policy_minus)
	C_L_plus = averagecost_randompolicy(C_L,T,P,policy_plus)
	C_L_minus = averagecost_randompolicy(C_L,T,P,policy_minus)
	C_L_avg = averagecost_randompolicy(C_L,T,P,policy)
	C_A_avg = averagecost_randompolicy(C_A,T,P,policy)
	del_C_A = np.zeros(pertub.shape[0])
	del_C_L = np.zeros(pertub.shape[0])

	for j in range(pertub.shape[0]):
	  if pertub[j]!=0:
		del_C_A[j] = (C_A_plus - C_A_minus)/(pertub[j]*delta[i])
		del_C_L[j] = (C_L_plus - C_L_minus)/(pertub[j]*delta[i])
	assert del_C_A.shape == parameters.shape
	parameters = parameters - epsilon[i]*(del_C_A + del_C_L*np.max([0, rho*(C_L_avg-D) ]))
	lamb = np.max([(1-epsilon[i]/rho)*lamb, lamb + epsilon[i]*(C_L_avg - D)])
	if i%100==0:
		print(i,C_A_avg)
	parameters_store[i] = parameters
	tau = 0.999*tau
	print(C_A_avg,C_L_avg-D,lamb,parameters)
	return parameters_store
#### Parameters
D = 0.6
parameters_initial = np.append(np.tile([7,12],O),np.pi/2)
n_iter = 6000
delt = np.linspace(0.5,0.4,n_iter)
T = 1000
lamb = 1
epsilon = np.linspace(0.4,0,n_iter)
rho = 2
parameters_spsa = spsa(parameters_initial,delt,n_iter,T,P,D,lamb,epsilon,rho,L,O,A,C_A,C_L)
#### Plotting Fig 2
n_iter_sample = 6000
plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,0],label="$y^O = O_1$")
plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,2],label="$y^O = O_2$")
plt.plot(np.arange(n_iter_sample),parameters_spsa[:n_iter_sample,4],label="$y^O = O_3$")
plt.legend(fontsize=12)
plt.legend(fontsize=12)
plt.xlabel("Iterations",size = 16)
plt.ylabel("Threshold Parameter of $y^L$", size = 16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.savefig("fig2.pdf",bbox_inches='tight',pad_inches=0)
plt.show()
