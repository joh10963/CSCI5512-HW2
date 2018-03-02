"""
    Code for Problem 2 - Gibbs Sampling
"""

import numpy as np

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
    # First, check to make sure CPT is correctly described by X and Y
    if CPT.ndim != (np.sum(X) + np.sum(Y)):
        print('X and Y are not consistent with CPT')
        return 0.0
    # Check if there is not enough evidence
    if e[np.argwhere(X)[0,0]] < 0 or np.any(e[np.argwhere(Y)[:, 0]] < 0):
        print('e does not contain enough evidence for CPT')
        return 0.0
    # find the values to use as evidence (to input into CPT) in the correct order
    vals = [e[i] for i in range(Y.size) if Y[i] == 1]
    # Input the vals into CPT
    output = CPT[e[np.argwhere(X)[0,0]]] #query variable X first
    for i in vals: # input all the rest of the variables
        output = output[i]

    return output

def get_indv_markov_prob(X, e, CPTs, G):
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
    # Get P(X=x|Parents(X))
    i = np.argwhere(X)[0,0]
    children_X = get_children(G, i)
    parents_X = get_parents(G, i)
    prob = get_prob(X, parents_X, e, CPTs[i])
    
    # Get P(Y=y|Parents(Y)) for each child, Y, of X
    for i in np.argwhere(children_X)[:, 0]:
        # create the variable vector
        y = np.zeros(G.shape[0])
        y[i] = 1
        # get the parents
        parents = get_parents(G, i)
        # add the product
        prob *= get_prob(y, parents, e, CPTs[i])

    return prob

def get_markov_prob(X, e, CPTs, G):
    '''
        Return the conditional probability P(X|Y1, Y2, ...) given evidence e for all values of X
        
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
            np.array
    '''
    i = np.argwhere(X)[0,0]
    prob = np.zeros(CPTs[i].shape[0])
    
    for x in range(prob.size):
        e[i] = x #change the evidence
        prob[x] = get_indv_markov_prob(X, e, CPTs, G)
    
    return prob / np.sum(prob)



def get_prob_string(variables, values, probs, X, e):
    '''
        Return the string P(X=e|Y=e) = z
        
        Inputs:
            variables: np.array, name of variables
            values: np.array, values of variables
            probs: np.array, probabilities to display
            X: np.array, one hot coding for query variable X (for CPT)
                X[i] = 1 if query variable is variable i
                     = 0 otherwise
            e: np.array, values of variables X and Y
                e[i] = val if variable i has evidence
                     = -1 if variable i has no evidence
        
        Outputs:
            string
    '''
    i = np.argwhere(X)[0,0]
    conditioned = [(variables[j],values[j][e[j]]) for j in range(e.size) if (e[j] >= 0 and j != i)]
    
    output = []
    for k in range(probs.size):
        s = 'P(%s = %s |' % (variables[i], values[i][k]) #query variable
        for pair in conditioned:
            s += '%s = %s, ' % pair #conditioned variables
        s = s[:-2] #slice off the extra space and comma
        s += ') = %s' % probs[k]
        output.append(s)

    return output

def gibbs_sample(X, e, G, CPTs, n):
    '''
        Return the estimate of probability P(X|e) using gibbs sampling algorithm
        
        Inputs:
        
        Outputs:
    '''
    # initilize a counter N to count the samples
    i = np.argwhere(X)[0,0]
    N = np.zeros(CPTs[i].shape[0])
    
    # nonevidence variables
    non = np.argwhere(e < 0)[:, 0] #get the indicies
    # one hot coding for variables
    Z = [(non[z], np.zeros(G.shape[0])) for z in range(non.size)]
    for ind, var in enumerate(non):
        Z[ind][1][var] = 1
    
    # initialize state x with random values for non evidence variables
    x = np.copy(e)
    for j in range(x.size):
        if x[j] == -1: # a non evidence variable
            x[j] = np.random.choice(CPTs[j].shape[0], 1, replace=False)

    # Start sampling!!!!!!
    for k in range(n):
        for (j, z) in Z:
            p = get_markov_prob(z, x, CPTs, G)
            x[j] = np.random.choice(CPTs[j].shape[0], 1, replace=False, p=p)
            N[x[i]] += 1

    return N / np.sum(N)

if __name__ == "__main__":
    
    variables = ['C', 'S', 'R', 'W']
    
    C = np.array(['T', 'F'])
    S = np.array(['T', 'F'])
    R = np.array(['N', 'M', 'F'])
    W = np.array(['T', 'F'])
    values = np.array([C, S, R, W], dtype=object)
    
    # Graph
    G = np.array([[0, 1, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    
    # Conditional Probability Tables
    Pc = np.array([0.5, 0.5])
    Ps = np.array([[0.1, 0.5],
                   [0.9, 0.5]])
    Pr = np.array([[0.6, 0.1],
                   [0.2, 0.1],
                   [0.2, 0.8]])
    Pw = np.array([[[0.99, 0.9, 0.9], [0.9, 0.7, 0.01]],
                   [[0.01, 0.1, 0.1], [0.1, 0.3, 0.99]]])
                   
    CPTs = [Pc, Ps, Pr, Pw]

    # The evidence given in the problem is S = T and W = T
    e = np.array([-1, 0, -1, 0])

    ###############################################################
    # Part b
    ###############################################################
    print('----------------------- Part b ----------------------')
    # Print the conditional probabilities of all the combinations of C and R
    for i, j in [(2, 0), (0, 2)]: #iterate between C and R for query varaible
        # create on-hot query variable
        X = np.zeros(G.shape[0])
        X[i] = 1
        for y in range(values[j].size): #iterate through values of non-query variable
            # update the evidence
            e[j] = y
            # calculate the probability
            p = get_markov_prob(X, e, CPTs, G)
            print('\n'.join(get_prob_string(variables, values, p, X, e)))
            print('-----------------------------------------------------')

    ###############################################################
    # Part c
    ###############################################################
    print('----------------------- Part c ----------------------')
    # The evidence given in the problem is S = T and W = T
    e = np.array([-1, 0, -1, 0])
    # The query variable is R
    X = np.array([0, 0, 1, 0])

    print('100 steps: ')
    p = gibbs_sample(X, e, G, CPTs, 100)
    print('\n'.join(get_prob_string(variables, values, p, X, e)))

    print('10000 steps: ')
    p = gibbs_sample(X, e, G, CPTs, 10000)
    print('\n'.join(get_prob_string(variables, values, p, X, e)))

    # The evidence given in the problem is S = T and W = F
    e = np.array([-1, 0, -1, 1])

    print('100 steps: ')
    p = gibbs_sample(X, e, G, CPTs, 100)
    print('\n'.join(get_prob_string(variables, values, p, X, e)))

    print('10000 steps: ')
    p = gibbs_sample(X, e, G, CPTs, 10000)
    print('\n'.join(get_prob_string(variables, values, p, X, e)))










