"""
    Code for Problem 2 - Gibbs Sampling
"""

import numpy as np
import pdb

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

def get_children(G, i):
    '''
        Return the children of node i in Bayes Net G
        
        Inputs:
            G: np.ndarray, an adjacency matrix describing the directed Bayes Net
                G[i, j] = 1 if there exists and edge from node i to j
                        = 0 otherwise
            i: integer, the index of the variable in question
        
        Outputs:
            C: np.array, a list of the parents of i in one-hot coding
                C[k] = 1 if k is a child of i
                     = 0 otherwise
        '''
    return G[i, :]

def get_prob(X, Y, e, CPT):
    '''
        Return the conditional probability P(X|Y1, Y2, ...) given evidence e
        
        Inputs:
            X: np.array, one hot coding for query variable X (for CPT)
                X[i] = 1 if query variable is variable i
                     = 0 otherwise
            Y: np.array, one hot coding for conditioned variables Y (for CPT)
                Y[i] = 1 if variable i is a conditioned variable
                     = 0 otherwise
            e: np.array, values of variables X and Y
                e[i] = val if variable i has evidence
                     = -1 if variable i has no evidence
            CPT: np.ndarray, conditional probability table P(X|Y1, Y2, ...)
                CPT[X=x, Y1=y1, Y2=y2, ...] ---> P(X=x | Y1=y1, Y2=y2, ...)
                CPT.ndim == np.sum(X) + np.sum(Y)
                
        Outputs:
            float
    '''
    #pdb.set_trace()
    # First, check to make sure CPT is correctly described by X and Y
    if CPT.ndim != (np.sum(X) + np.sum(Y)):
        print('X and Y are not consistent with CPT')
        return 0.0
    # Check if there is not enough evidence
    if e[np.argwhere(X)[0,0]] < 0 or np.any(e[np.argwhere(Y)[:, 0]] < 0):
        print('e does not contain enough evidence for CPT')
        return 0.0
    # find the values to use as evidence (to input into CPT) in the correct order
    vals = [e[np.argwhere(X)[0,0]]] + [e[i] for i in range(Y.size) if Y[i] == 1]
    # Input the vals into CPT
    output = CPT[vals[0]] #query variable X first
    for i in vals[1:]: # input all the rest of the variables
        output = output[i]

    return output

def get_markov_prob(X, e, CPTs, G):
    '''
        Return the conditional probability P(X|Y1, Y2, ...) given evidence e
        
        Inputs:
            X: np.array, one hot coding for query variable X (for CPT)
                X[i] = 1 if query variable is variable i
                     = 0 otherwise
            e: np.array, values of variables X and Y
                e[i] = val if variable i has evidence
                     = -1 if variable i has no evidence
            G: np.ndarray, an adjacency matrix describing the directed Bayes Net
                G[i, j] = 1 if there exists and edge from node i to j
                        = 0 otherwise
        CPTs: list of np.ndarray, conditional probability table P(X|Y1, Y2, ...)
            CPTs[i] = CPT[X=x, Y1=y1, Y2=y2, ...] ---> P(X=x | Y1=y1, Y2=y2, ...)
            CPT.ndim == np.sum(X) + np.sum(Y)
        
        Outputs:
            float
    '''
    i = np.argwhere(X)[0,0]
    parents_X = get_parents(G, i)
    prob = get_prob(X, parents_X, e, CPT[i])

    print prob


def get_prob_string(variables, values, X, Y, e, CPT):
    '''
        Return the string P(X=e|Y=e) = z
        
        Inputs:
            variables: np.array, name of variables
            values: np.array, values of variables
            X: np.array, one hot coding for query variable X (for CPT)
                X[i] = 1 if query variable is variable i
                     = 0 otherwise
            Y: np.array, one hot coding for conditioned variables Y (for CPT)
                Y[i] = 1 if variable i is a conditioned variable
                     = 0 otherwise
            e: np.array, values of variables X and Y
                e[i] = val if variable i has evidence
                     = -1 if variable i has no evidence
            CPT: np.ndarray, conditional probability table P(X|Y1, Y2, ...)
                CPT[X=x, Y1=y1, Y2=y2, ...] ---> P(X=x | Y1=y1, Y2=y2, ...)
                CPT.ndim == np.sum(X) + np.sum(Y)
        
        Outputs:
            string
    '''
    prob = get_prob(X, Y, e, CPT) #We probably want to change this to markov prob
    
    i = np.argwhere(X)[0,0]
    query = (variables[i], values[i][e[i]])
    conditioned = [(variables[i],values[i][e[i]]) for i in range(Y.size) if Y[i] == 1]
    
    output = 'P(%s = %s |' % query
    for pair in conditioned:
        output += '%s = %s, ' % pair
    output = output[:-2] #slice off the extra space and comma
    output += ') = %s' % prob

    return output



if __name__ == "__main__":
    
    variables = ['C', 'S', 'R', 'W']
    
    C = np.array(['T', 'F'])
    S = np.array(['T', 'F'])
    R = np.array(['N', 'M', 'F'])
    W = np.array(['T', 'F'])
    values = np.array([C, S, R, W], dtype=object)
    
    # Graph
    G = np.array([[0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1]])
    
    # Conditional Probability Tables
    Pc = np.array([0.5, 0.5])
    Ps = np.array([[0.1, 0.5],
                   [0.9, 0.5]])
    Pr = np.array([[0.6, 0.1],
                   [0.2, 0.1],
                   [0.2, 0.8]])
    Pw = np.array([[[0.99, 0.9, 0.9], [0.9, 0.7, 0.01]],
                   [[0.01, 0.1, 0.1], [0.1, 0.3, 0.99]]])

    X = np.array([0, 0, 0, 1])
    Y = np.array([0, 1, 1, 0])
    e = np.array([-1, 0, 0, 1])

    print(get_prob_string(variables, values, X, Y, e, Pw))











