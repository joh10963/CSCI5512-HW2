"""
 Code for HW2 Problem 3
"""

import numpy as np
import pdb

from aima import probability as pb

################################################
# Part b - inference by enumeration from scratch
################################################
def get_parents(G, i):
    '''
        Return the parents of node i in Bayes Net G
        
        Inputs:
            G: np.ndarray, an adjacency matrix describing the directed Bayes Net
                G[i, j] = 1 if there exists and edge from node i to j
                        = 0 otherwise
            i: integer, the index of the variable in question
        
        Outputs:
            P: np.array, a list of the parents of i in one-hot coding
                P[k] = 1 if k is a parent of i
                     = 0 otherwise
        '''
    return G[:, i]

def get_prob(G, Y, vals, CPT):
    #pdb.set_trace()
    parents = get_parents(G, np.argwhere(Y)[0,0])
    parent_evidence = [vals[i] for i in range(vals.size) if parents[i] == 1]
    return_value = CPT[vals[np.argwhere(Y)[0,0]]] # evidence for Y
    for i in parent_evidence:
        return_value = return_value[i]
    return return_value

def enumeration_ask(X, vals, CPT, G):
    Q = np.zeros(shape=[CPT[np.argwhere(X)[0,0]].shape[0]])
    vars = np.identity(G.shape[0])
    for x in range(Q.size):
        vals[np.argwhere(X)[0,0]] = x
        Q[x] = enumerate_all(vars, vals, CPT, G)
    
    return Q/np.sum(Q)

def enumerate_all(vars, vals, CPT, G):
    if len(vars) == 0:
        return 1.0
    Y = vars[0]
    # if Y is an evidence variable then we don't have to sum over the variables
    if np.any(np.argwhere(Y) == np.argwhere(vals >= 0)):
        prob = get_prob(G, Y, vals, CPT[np.argwhere(Y)[0,0]])
        #pdb.set_trace()
        part = prob * enumerate_all(vars[1:], vals, CPT, G)
    else: # Y is not an evidence variable so sum over the variables
        part = []
        for y in range(CPT[np.argwhere(Y)[0,0]].shape[0]): #iterate through all values of Y
            # make y an evidence variable
            vals_prime = np.copy(vals)
            vals_prime[np.argwhere(Y)[0,0]] = y
            prob = get_prob(G, Y, vals_prime, CPT[np.argwhere(Y)[0,0]])
            #pdb.set_trace()
            part.append(prob * enumerate_all(vars[1:], vals_prime, CPT, G))
        part = sum(part)
    return part

###################################################################################
# Part c - inference by enumeration using AIMA
###################################################################################
def create_aima_net(chief_signal_inducing_crime_p):
    # Create the Bayes Net
    net = pb.BayesNet([('S', '', chief_signal_inducing_crime_p),
                       ('V', 'S', {(True):0.9, (False):0.2}),
                       ('C', 'S', {(True):0.98, (False):0.02}),
                       ('A', 'C', {(True):0.999, (False):0.0})])
    return net


if __name__ == '__main__':
####################################################################################
# Part b
####################################################################################
    # Define the Bayes Net G
    variable_names = ['S', 'V', 'C', 'A']
    G = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

    # P(S)
    # Ps = np.array([P(S=1), P(S=2), P(S=3), P(S=4), P(S=5)])
    S = [1, 2, 3, 4, 5]
    Ps = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
  
    # P(V|S) P[V, S]
    # Pv = np.array([[P(V=1|S=1) P(V=1|S=2) P(V=1|S=3) P(V=1|S=4) P(V=1|S=4)],
    #               [P(V=2|S=1) P(V=2|S=2) P(V=2|S=3) P(V=2|S=4) P(V=2|S=4)],
    #               [P(V=3|S=1) P(V=3|S=2) P(V=3|S=3) P(V=3|S=4) P(V=3|S=4)],
    #               [P(V=4|S=1) P(V=4|S=2) P(V=4|S=3) P(V=4|S=4) P(V=4|S=4)],
    #               [P(V=5|S=1) P(V=5|S=2) P(V=5|S=3) P(V=5|S=4) P(V=5|S=4)]])
    Pv = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                   [0.1, 0.8, 0.1, 0.0, 0.0],
                   [0.0, 0.1, 0.8, 0.1, 0.0],
                   [0.0, 0.0, 0.1, 0.8, 0.1],
                   [0.0, 0.0, 0.0, 0.1, 0.9]])
  
    # P(C|V) P[C, V]
    #Pc = np.array([[P(C=T|V=1), P(C=T|V=2), P(C=T|V=3), P(C=T|V=4), P(C=T|V=5)],
    #               [P(C=F|V=1), P(C=F|V=2), P(C=F|V=3), P(C=F|V=4), P(C=F|V=5)]])
    Pc = np.array([[0.02, 0.02, 0.02, 0.98, 0.98],
                 [0.98, 0.98, 0.98, 0.02, 0.02]])
  
    # P(A|C) P[A, C]
    #Pa = np.array([[P(A=T|C=T), P(A=T|C=F)],
    #               [P(A=F|C=T), P(A=F|C=F)]])
    Pa = np.array([[0.999, 0.0],
                 [0.001, 1.0]])
  
    CPT = [Ps, Pv, Pc, Pa]

    X = np.array([1, 0, 0, 0])
    vals = np.array([-1, -1, -1, 0])
    P = enumeration_ask(X, vals, CPT, G)
    print('------------ inference by enumeration from scratch ---------------')
    print('P(S|A=T) = %s' % P)
    print('P(S=5|A=T) = %s' % P[4])

##############################################################
# Part c
##############################################################
    chief_signal_inducing_crime_p = np.sum(P[3:]) # P(S>=4|A=T)
    net = create_aima_net(chief_signal_inducing_crime_p)
    
    # find P(S>=4|A=T) ----> P(S=T|A=T)
    X = 'S' #query variable
    e = {'A':True} # evidence
    P = pb.enumeration_ask(X, e, net) #P(S|A=T)
    
    print('------------ inference by enumeration from AIMA (binary) ---------------')
    print('P(S|A=T) %s' % P.show_approx())
    print('P(S>=4|A=T) %s' % P[True])

##############################################################
# Part d
##############################################################
    # can use the same net and variables from part c
    P2 = pb.elimination_ask(X, e, net) #P(S|A=T)

    print('------------ inference by variable elimination from AIMA (binary) ---------------')
    print('P(S|A=T) %s' % P2.show_approx())
    print('P(S>=4|A=T) %s' % P2[True])


## Define the Bayes Net G
## The order of the variables are [B, E, A, J, M]
#    variable_names = ['B', 'E', 'A', 'J', 'M']
#    G = np.array([[0, 0, 1, 0, 0],
#                  [0, 0, 1, 0, 0],
#                  [0, 0, 0, 1, 1],
#                  [0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0]])
#        
## Define the CPT for each variable in G
#  # P(b)
#  # Pb = np.array([-b, b])
#    Pb = np.array([0.001, 1.0-0.001])
#  
#  # P(E)
#  # Pe = np.array([-e, e])
#    Pe = np.array([0.002, 1.0-0.002])
#  
#  # P(A|B,E) P[A, B, E]
#  #Pa = np.array([[(-b, -e, -a), (-b, -e, a)], [(-b, e, -a), (-b, e, a)]],
#  #               [[(b, -e, -a), (b, -e, a)], [(b, e, -a), (b, e, a)]])
#    Pa = np.array([[[0.95, 0.94], [0.29, 0.001]],
#                   [[0.05, 0.06], [0.71, 0.999]]])
#  
#  # P(J|A) P[J, A]
#  #Pj = np.array([(-a, -j), (-a, j)],
#  #              [(a, -j), (a, j)])
#    Pj = np.array([[0.90, 0.05],
#                   [0.10, 0.95]])
#  
#  # P(M|A) P[M, A]
#  #Pm = np.array([(-a, -m), (-a, m)],
#  #              [(a, -m), (a, m)])
#    Pm = np.array([[0.70, 0.01],
#                   [0.30, 0.99]])
#  
#    CPT = [Pb, Pe, Pa, Pj, Pm]
#    
#    X = np.array([1, 0, 0, 0, 0])
#    vals = np.array([-1, -1, -1, 0, 0])
#    print enumeration_ask(X, vals, CPT, G)











