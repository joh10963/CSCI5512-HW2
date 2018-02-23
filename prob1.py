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
            R: np.ndarray, one hot coding for the variables in each factor
                R[i, j] = 1 if factor i uses variable j
                        = 0 otherwise
            CPT: list
    '''
    # start off with a factor graph the same size as G - add factor nodes as we go
    F = np.zeros(shape=G.shape)
    R = np.zeros(shape=G.shape) #factors x variables
    
    # Iterate through each node in G
    for i in range(G.shape[0]):
        # add a factor to the graph
        F = np.concatenate([F, np.zeros(shape=(F.shape[0],1))], axis=1) #add a column of zeros
        F = np.concatenate([F, np.zeros(shape=(1,F.shape[1]))], axis=0) #add a row of zeros
        R = np.concatenate([R, np.zeros(shape=(1,R.shape[1]))], axis=0) #add a row of zeros
        j = F.shape[0]-1 #the index of the factor we just added
        if np.argwhere(get_parents(G, i)).size > 0: # node i has parents so gotta connect those first
            for p in np.argwhere(get_parents(G, i)):
                # connect the parent to the factor
                F[p, j] = -1 # j is the factor so -1
                F[j, p] = 1 # p is the variable so 1
                R[j, p] = 1
        # Connect the factor to node i
        F[i, j] = -1 # j is the factor so -1
        F[j, i] = 1 # i is the variable so 1
            
        R[j, i] = 1
        CPT.append(CPT[i])
    
    return F, R, CPT

def not_sum(rep, ex, fCPT, m=None):
    '''
        Return the not-sum function over all variables except n of the product of fCPT and M
        
        Inputs:
            ex: np.array, shows the exclusion variable in one-hot coding
                ex[i] = 1 if variable i is exclusion variable
                      = 0 otherwise
            rep: np.array, shows what variables are in the fCPT
                rep[i] = 1 if variable i is in fCPT
                       = 0 otherwise
            fCPT: np.ndarray, CPT
            m: list of np.ndarray message arrays
        
        Outputs:
            output: np.ndarray
    '''
    # element-wise multiplication of the CPT and the messages
    if m is not None:
        output = np.multiply(fCPT, m[0])
        for i in range(1, len(m)):
            output = np.multiply(output, m[i])
    else:
        output = fCPT

    # find the the index of the left-out variable (in terms of the indexing of fCPT)
    #import pdb; pdb.set_trace()
    n = np.sum(rep[0:np.argwhere(ex)[0][0]])

    # sum over all the dimensions besides n
    sum_dim = [0 if d < n else 1 if d > n else -1 for d in range(fCPT.ndim)]
    sum_dim = [d for d in sum_dim if d >= 0]
    for dim in sum_dim:
        output = np.sum(output, axis=dim)

    return output

def get_incoming_messages(n, M):
    '''
        Return a list of the incoming messages to node n
        
        Inputs:
            n: integer, index of node
            M: np.ndarray, an adjacency matrix describing the message graph
                M[i, j] = np.ndarray if there is a message from node i to j
                        = None otherwise
        
        Outputs:
            m: list of np.ndarray
                m[i] = np.ndarray if node i sends a message to n
                     = None if node i does not send a message to n
    '''
    incoming_messages = M[:, n] #messages going from somewhere to sending_node
    # find the messages (can't use built in numpy functions for this)
    messages = incoming_messages.shape[0] * [None]
    for i in range(incoming_messages.shape[0]):
        message = incoming_messages[i]
        if not np.any(np.isnan(message)): #it's a message
            messages[i] = message

    return messages


def find_ends(F):
    '''
        Return a list of node indices of the ends to start the message passing
        
        Inputs:
            F: np.ndarray, an adjacency matrix describing the factor graph of G
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
                        
        Outputs:
            indices: list
                indices[i] = 1 if node i is an end and a variable
                           = -1 if node i is an end and a factor
                           = 0 otherwise
    '''
    ends = np.zeros(shape=(F.shape[0]))
    
    for i in range(F.shape[0]):
        if np.argwhere(F[:, i]).size == 1: #node i has one parent (b/c undirected graph)
            ends[i] = np.sum(F[:, i])
        if np.argwhere(F[i, :]).size == 1: #node i has one children (b/c undirected graph)
            ends[i] = np.sum(F[:, i])

    return ends

def get_initial_message_matrix(F, R, CPT):
    '''
        Return the initialization of the message adjacency matrix
    '''
    ends = find_ends(F) #find the nodes that are ends - where our messages start
    M = np.full(F.shape, np.nan, dtype=object) #create an array of size F filled with np.nan
    
    # if the ends are variables then start with a message of 1
    for i, e in enumerate(ends):
        if e == 1: #it's a variable
            children = np.argwhere(get_children(F, i))[0][0]
            M[i,  children] = np.array([1.0])
        elif e == -1: #it's a factor
            c = get_children(F, i)
            c_index = np.argwhere(c)[0][0]
            M[i,  c_index] = not_sum(R[i,:], c, CPT[i])

    return M


def calculate_message(F, R, M, CPT, sending_node, receiving_node):
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
            R: np.ndarray, one hot coding for the variables in each factor
                R[i, j] = 1 if factor i uses variable j
                        = 0 otherwise
            CPT: np.ndarray, CPT for all factors
            sending_node: integer, index of node in F sending the message
            receiving_node: integer, index of node in F receiving the message
        
        Outputs:
            message: np.array
    '''
    # get the incoming messages to the sending_node - excluding any messages from the receiving node
    messages = get_incoming_messages(sending_node, M)
    incoming_messages = [messages[i] for i in range(len(messages)) if (i != receiving_node and messages[i] is not None)]
    
    # if node is a variable, then just multiply all the incoming messages together
    if F[receiving_node, sending_node] == 1:
        message = incoming_messages[0]
        for i in range(1, len(incoming_messages)):
            message = np.multiply(message, incoming_messages[i])
    
    # if node is a factor, then do the not_sum of the factor and the incoming messages
    else:
        exclude = np.zeros(shape=(R.shape[1]))
        exclude[receiving_node] = 1
        message = not_sum(R[sending_node,:], exclude, CPT[sending_node])

    return message

def get_node_order(F):
    output = []
        
    # start with the end nodes
    send_nodes = np.argwhere(find_ends(F))[:,0]
    visited = send_nodes
    
    # forward pass
    while visited.size < F.shape[0]: #while we haven't visited all the nodes
        #update the visited list
        visited = np.concatenate([visited, send_nodes])
        visited = np.unique(visited)
        
        # find the next nodes to evaluate
        receiving = [np.argwhere(get_children(F, node))[:,0] for node in send_nodes] #the children of node are the next to be evaluated
        receiving = [[e for e in children if e not in visited] for children in receiving] #but we only want ones we haven't visited yet
        if len(receiving[0]) == 0:
            receiving = [[send_nodes[i] for i in range(send_nodes.size) if send_nodes[i] != send_nodes[j]] for j in range(send_nodes.size)] # for when we do the cross over
    
        # append this step
        output.append({send:receive for (send, receive) in [(send_nodes[i], receiving[i]) for i in range(len(send_nodes))]})

        # update the send_nodes list
        send_nodes = np.unique([r for x in receiving for r in x]) # the nodes visited in this step
    
    # backward pass
    length = len(output)-2 #start at the second to last item in output
    for i in range(length, -1, -1):
        send_nodes = np.unique(output[i].values())
        receiving = [[key for key, value in output[i].items() if r in value] for r in send_nodes]
        output.append({send:receive for (send, receive) in [(send_nodes[i], receiving[i]) for i in range(len(send_nodes))]})

    return output

def pass_messages(F, R, CPT):
    '''
        Returns the completed message graph after passing all the messages
        
        Inputs:
            F: np.ndarray, an adjacency matrix describing the factor graph
                F[i, j] = 1 if there exists an edge from node i to j and node j is a variable
                        = -1 if there exists an edge from node i to j and node j is a factor
                        = 0 otherwise
            R: np.ndarray, one hot coding for the variables in each factor
                R[i, j] = 1 if factor i uses variable j
                        = 0 otherwise
            CPT: np.ndarray, CPT for all factors
        
        Outputs:
            M: np.ndarray, an adjacency matrix describing the message graph
                M[i, j] = np.ndarray if there is a message from node i to j
                        = None otherwise
    '''
    # initialize M with first messages passed
    M = get_initial_message_matrix(F, R, CPT)
    order = get_node_order(F)
    for step in order[1:]: #skip the first step b/c of how we initialized M
        # step is {send_node1:[receiving_nodes], ...}
        # pass the message for nodes each step
        for send_node,receiving_nodes in step.items():
            for r in receiving_nodes:
                message = calculate_message(F, R, M, CPT, send_node, r)
                # save the message in M
                M[send_node, r] = message

    return M


def get_marginal_distributions(G, F, M):
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
    P = G.shape[0]*[None]
    for var in range(G.shape[0]):
        # get the incoming messages to the var node
        messages = get_incoming_messages(var, M)
        messages = [messages[i] for i in range(len(messages)) if (messages[i] is not None)]

        # multiply all the incoming messages together
        message = messages[0]
        for i in range(1, len(messages)):
            message = np.multiply(message, messages[i])
        
        # save the probability distribution (normalize)
        P[var] = message / np.sum(message)

    return P

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
    F, R, CPT = create_factor_graph(G)
    # pass the messages
    M = pass_messages(F, R, CPT)
    # get the marginal distributions
    P = get_marginal_distributions(G, F, M)
    
    return P

if __name__ == '__main__':

    # Define the Bayes Net G
    # The order of the variables are [B, E, A, J, M]
    variable_names = ['B', 'E', 'A', 'J', 'M']
    G = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    
    # Define the CPT for each variable in G
    # P(b)
    # Pb = np.array([-b, b])
    Pb = np.array([1.0-0.001, 0.001])

    # P(E)
    # Pe = np.array([-e, e])
    Pe = np.array([1.0-0.002, 0.002])

    # P(A|B,E) P[B, E, A]
    #Pa = np.array([[(-b, -e, -a), (-b, -e, a)], [(-b, e, -a), (-b, e, a)]],
    #               [[(b, -e, -a), (b, -e, a)], [(b, e, -a), (b, e, a)]])
    Pa = np.array([[[1.0-0.001, 0.001], [1.0-0.29, 0.29]],
                   [[1.0-0.94, 0.94], [1.0-0.95, 0.95]]])

    # P(J|A) P[A, J]
    #Pj = np.array([(-a, -j), (-a, j)],
    #              [(a, -j), (a, j)])
    Pj = np.array([[1.0-0.05, 0.05],
                   [1.0-0.90, 0.90]])
    
    # P(M|A) P[A, M]
    #Pm = np.array([(-a, -m), (-a, m)],
    #              [(a, -m), (a, m)])
    Pm = np.array([[1.0-0.01, 0.01],
                   [1.0-0.70, 0.70]])
    
    CPT = [Pb, Pe, Pa, Pj, Pm]

    P = belief_propagation(G, CPT)
    for i in range(len(variable_names)):
        print 'Marginal Distribution ([false, true]) for variable %s is  %s' %(variable_names[i], P[i])


#print not_sum(np.array([1, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0]), Pb, [np.array([0.1, 0.2])])
#print not_sum(np.array([0, 1, 0, 0, 0]), np.array([0, 1, 0, 0, 0]), Pe, [np.array([0.1, 0.2])])
#print not_sum(np.array([1, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0]), Pa, [np.array([0.1, 0.2])])
#print not_sum(np.array([0, 0, 1, 1, 0]), np.array([0, 0, 0, 1, 0]), Pj, [np.array([1])])

















