"""
    CSCI5512 HW2 Problem 2
    
    Calculates the marginal distributions of each variable given:
        1) a Bayes Net
        2) CPT for each variable in Bayes Net
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

def create_factor_graph(G):
    '''
        Return a factor graph given the Bayes Net G
        
        Inputs:
            G: np.ndarray, an adjacency matrix describing the directed Bayes Net 
                G[i, j] = 1 if there exists and edge from node i to j
                        = 0 otherwise
            
        Outputs:
            F: np.ndarray, an adjacency matrix describing the factor graph of G
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
    '''
    pass

def not_sum(n, fCPT, M):
    '''
        Return the not-sum function over all variables except n of the product of fCPT and M
        
        Inputs:
            n: integer, index of the exception variable in fCPT
            fCPT: np.ndarray, CPT
            M: list of np.ndarray message arrays
        
        Outputs:
            sum: np.ndarray
    '''
    pass

def find_ends(F):
    '''
        Return a list of node indices of the ends to start the message passing
        
        Inputs:
            F: np.ndarray, an adjacency matrix describing the factor graph of G
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
                        
        Outputs:
            indices: np.array, indices of end nodes of F
    '''
    pass

def calculate_message(F, M, CPT, node):
    '''
        Return the message vector of the node in factor graph F
        
        Inputs:
            F: np.ndarray, an adjacency matrix describing the factor graph
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
            M: np.ndarray, an adjacency matrix describing the message graph
                M[i, j] = np.ndarray if there is a message from node i to j
                        = None otherwise
            CPT: np.ndarray, CPT for all factors
            node: integer, index of node in F
        
        Outputs:
            message: np.array
    '''
    # if node is a variable, do this calculation
    # if node is a factor, do this calculation
    pass

def pass_messages(F, CPT):
    '''
        Returns the completed message graph after passing all the messages
        
        Inputs:
            F: np.ndarray, an adjacency matrix describing the factor graph
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
        CPT: np.ndarray, CPT for all factors
        
        Outputs:
            M: np.ndarray, an adjacency matrix describing the message graph
                M[i, j] = np.ndarray if there is a message from node i to j
                        = None otherwise
    '''
    # initialize M, with [ones] for edges
    # initialize current nodes list - find_ends(F)
    # for i = 1 to n
    #   for node in current node list
    #       calculate_message(F, M, CPT, node)
    #       save message in correct spot in M
    #       update the list of current nodes with the next node to calculate
    pass

def get_marginal_distributions(F, M):
    '''
        Returns the marginal distributions of the variables in F using the completed message graph M
        
        Inputs:
            F: np.ndarray, an adjacency matrix describing the factor graph
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
            M: np.ndarray, an adjacency matrix describing the message graph
                M[i, j] = np.ndarray if there is a message from node i to j
                        = None otherwise
        
        Outputs:
            P: np.ndarray, marginal distributions for each variable in F
    '''
    # for each variable in F
    #   calculate the marginal probability of that variable (the product of incoming messages)
    pass

def belief_propagation(G, CPT):
    '''
        Returns the marginal distributions for the variables in Bayes Net G
        
        Inputs:
            G: np.ndarray, an adjacency matrix describing the directed Bayes Net
                G[i, j] = 1 if there exists and edge from node i to j
                        = 0 otherwise
            CPT: np.ndarray, CPT for all factors
        
        Outputs:
            P: np.ndarray, marginal distributions for each variable in G
    '''
    # Create factor graph F
    # M = pass_messages(F, CPT)
    # P = get_marginal_distributions(F, M)
    # return P
    pass

if __name__ == '__main__':

    # Define the Bayes Net G
    # The order of the variables are [B, E, A, J, M]
    variable_names = ['B', 'E', 'A', 'J', 'M']
    G = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
                  
    CPT = [None, None, None, None, None] #these need to be np.ndarrays - not sure how to define quite yet


















