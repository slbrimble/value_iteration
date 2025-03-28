# Value_iteration Function Examples
## Party Example
---
This example is taken from Example 9.27 in Artificial Intelligence: Foundations and Computational Agents 2nd edition, which is linked below.
https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html.

A person needs to decide whether to party or rest. The possible states are 'healthy' and 'sick', and the possible actions are 'relax' or 'party'.

The transition probabilities are given by:  

$P(healthy\ |\ healthy, \ relax)=0.95$  

$P(healthy\ |\ healthy,\  party)=0.7$  

$P(healthy\ |\ sick,\  relax)=0.5$  

$P(\ healthy|\ sick,\  party)=0.1$.  


The rewards are given by:  

$R(healthy,\  relax)=7$  

$R(healthy,\ party)=10$  

$R(sick,\ relax)=0$  

$R(sick,\ party)=2$.

Load package


```python
import value_iteration
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[11], line 1
    ----> 1 import value_iteration
    

    ModuleNotFoundError: No module named 'value_iteration'


Define states:


```python
party_S = ['healthy', 'sick']
```

Define actions:


```python
party_A = ['relax', 'party']
```

Define transition probabilities:


```python
def party_P(s, a, s_dash):
    if s == 'healthy':
        if a == 'relax':
            if s_dash == 'healthy':
                return 0.95
            else:
                return 0.05
        else:
            if s_dash == 'healthy':
                return 0.7
            else:
                return 0.3
    else:
        if a == 'relax':
            if s_dash == 'healthy':
                return 0.5
            else:
                return 0.5
        else:
            if s_dash == 'healthy':
                return 0.1
            else:
                return 0.9
```

Define rewards:


```python
def party_R(s, a):
    if s == 'healthy':
        if a == 'relax':
            return 7
        else:
            return 10
    else:
        if a == 'relax':
            return 0
        else:
            return 2
```

Run Value_iteration function and observe outputted value function iterations and policy.


```python
party_vit = Value_iteration(S, A, P, R, gamma=0.8, K=1000)
print(party_vit[1])
print([party_vit[0][i][1000] for i in range(2)])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 1
    ----> 1 party_vit = Value_iteration(S, A, P, R, gamma=0.8, K=1000)
          2 print(party_vit[1])
          3 print([party_vit[0][i][1000] for i in range(2)])
    

    NameError: name 'Value_iteration' is not defined


## Pig example
---
This example emulates an adapted round of the game Pig. A player repeatedly rolls a die. If they roll a one then they get a score of zero, otherwise they add the number on the die to their score. Each time they roll, they have the option to stick with their score and stop rolling or roll again. If they reach a score of 100 then they have won the game (if the final roll brings the player to a score greater than 100, this is classed as 100). The possible actions are 'roll' or 'stick', and the possible states are $0, 1, \dots, 100$.

The reward function for state s is:  

$R(s,\ stick)=0$  

$R(s, \ roll)=(20-s)/6$  

(the reward for rolling is the expected score).

The transition probabilities are for $P(s+i|\ s,\ roll)=1/6$ for $i\in\{2,3,4,5,6\}$, $P(0|\ s,\ roll)=1/6$, and $P(s'|\ s,\ roll)=0$ otherwise (this is slightly different for $s'>94$, when the probability of getting 100 is actually the probability of getting 100 or higher. $P(s|\ s,\ stick)=1$ and $P(s'|\ s,\ stick)=0$ for $s'\neq s$.

Define states:


```python
pig_S = [i for i in range(101)]
```

Define actions:


```python
pig_A = ['roll', 'stick']
```

Define reward function


```python
def pig_R(s, a):
    if a == 'stick':
        return 0
    else:
        return (20-s)/6
```

Define transition probabilities:


```python
def pig_P(s, a, s_dash):
    if s > 94:
        if a == 'stick':
        if s_dash == s:
            return 1
        else:
            return 0
    else:
        diff = 100 - s
        if s_dash == 100:
            return 1 - diff / 6+1/6
        elif s_dash > s + 1:
            return 1/6
        elif s_dash == 0:
            return 1/6
        else:
            return 0
    
    if a == 'stick':
        if s_dash == s:
            return 1
        else:
            return 0
    else:
        if s_dash == 0:
            return 1/6
        elif s_dash - s <= 6 and s_dash - s >= 2:
            return 1/6
        else:
            return 0
```

Run Value_iteration function and observe outputted value function iterations and policy.


```python
pig_vit = Value_iteration(pig_S, pig_A, pig_P, pig_R, gamma=0.8, K=100)
print(pig_vit[1])
print([pig_vit[0][i][100] for i in range(2)])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 pig_vit = Value_iteration(pig_S, pig_A, pig_P, pig_R, gamma=0.8, K=100)
          2 print(pig_vit[1])
          3 print([pig_vit[0][i][100] for i in range(2)])
    

    NameError: name 'Value_iteration' is not defined

