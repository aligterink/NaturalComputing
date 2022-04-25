import numpy as np
from poibin import PoiBin
import math
import itertools
import matplotlib.pyplot as plt

def get_prob(p):
    pb = PoiBin(p)
    n = len(p)
    min_n =  int((n+1) / 2 + (n+1) % 2) # the minimal required votes for success
    total_prob = 0

    for t in range(min_n, n+1):
        # print(t, pb.pmf(t))
        total_prob += pb.pmf(t)
    print(total_prob)
    return total_prob

def exercise_3a():
    p = []
    p.extend([0.6]*10)
    p.extend([0.8])
    get_prob(p)

def exercise_3b():
    total_probs, w_08_list = [], []
    p = [0.6]*10
    p.append(0.8)
    p_reverse = [1-x for x in p]

    permu_list = list(itertools.product([False, True], repeat=11))
    
    for i in np.arange(0,1,0.01):
        w_08 = i
        w_06 = (1-i)/10

        w = [w_06]*10
        w.append(w_08)

        total_prob = 0

        for permu in permu_list:
            rel_w = [x for x, y in zip(w, permu) if y == True]
            if sum(rel_w) > 0.5:
                rel_p = [x for x, y in zip(p, permu) if y == True]
                rel_p_rev = [x for x, y in zip(p_reverse, permu) if y == False]
                total_prob += math.prod(rel_p) * math.prod(rel_p_rev)
        
        total_probs.append(total_prob)
        w_08_list.append(i)

    objects = [str(x)+'/11' for x in w_08_list]
    objects[0] = 0
    objects[11] = 1
    y_pos = np.arange(len(objects))

    plt.plot(y_pos, total_probs)
    xticks = ['0.'+str(x) for x in np.arange(0,10,1)]
    xticks.append('1')
    plt.xticks(np.arange(0, len(objects), len(objects)/11), xticks)
    plt.ylabel('Probability of correct majority vote')
    plt.xlabel('w')
    plt.title('Probability of correct majority vote for different strong classifier weights')

    plt.show()

    return total_probs, w_08_list

def exercise_3c():
    weights = []
    errors = np.arange(0, 1, 0.01)
    for e in errors:
        weights.append(1/2 * math.log((1-e)/e))
    plt.plot(weights)
    xticks = ['0.'+str(x) for x in np.arange(0, 10, 1)]
    xticks.append('1')
    print(xticks)
    plt.xticks(np.arange(0, len(errors)+1, 10), xticks)
    plt.xlabel('Error rate')
    plt.ylabel('Weight')
    plt.title('Assigned weight for different classifier error rates')
    plt.show()

exercise_3c()
