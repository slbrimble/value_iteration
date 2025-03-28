#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Calculates the Q-function for a given action a, state s, the set of states S, reward function R, transition probability P, V is the value function and
# k is the iteration, gamma is the discount factor.
def value(a, s, S, R, P, V, k, gamma):
    prob = [P(s, a, S[i]) for i in range(len(S))]
    val = [V[i][k-1] for i in range(len(V))]
    return R(s, a) + gamma*sum([prob[i]*val[i] for i in range(len(prob))])


# In[2]:


# Calculates the Q-function for each action in the list of actions A for a given state s and returns them in a list. R is the reward function, 
# P is the transition probability function, V is the value function, k is the current iteration, and gamma is the discount factor.
def A_value(s, A, S, R, P, V, k, gamma):
    return [value(a, s, S, R, P, V, k, gamma) for a in A]


# In[3]:


# Calculates value function for state s. A is the list of actions, S the list of states, R is the reward function, 
# P is the transition probability function, V is the value function, k is the current iteration, and gamma is the discount factor.
def max_value(s, A, S, R, P, V, k, gamma):
    return max(A_value(s, A, S, R, P, V, k, gamma))


# In[4]:


# Termination criteria for value iteration. k is current iteration and K is total number of iterations. Terminate if k>K.
def termination(k, K):
    if k > K:
        return True
    else:
        return False


# In[5]:


# A is the list of actions, S the list of states, R is the reward function, P is the transition probability function, K is the total number of 
# iterations, and gamma is the discount factor. Returns a list of the V, the list iterations of the value function for each state, and pi, the list of 
# policies for each state. 

def Value_iteration(S, A, P, R, gamma, K):
    k = 0
    V = [[0] for i in range(len(S))]
    pi = [ ]
    while termination(k+1, K) == False:
        k = k + 1
        i = 0
        for s in S:
            V[i].append(max_value(s, A, S, R, P, V, k, gamma))
            i = i + 1
    for s in S:
        ind = A_value(s, A, S, R, P, V, k, gamma).index(max_value(s, A, S, R, P, V, k, gamma))
        pi.append(A[ind])
    return [V, pi]     

