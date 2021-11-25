import math
import numpy as np
class treeNode:
    r_ij = 0
    pu = 0
    pm = 0
    pd = 0
    mi = 0
    Q = 0
    B = 0
    def __init__(self, i, j):
        self.i = i
        self.j = j
def i_to_arrayIndex(i,arr):
    Index = abs(i - arr[0].i)
    return Index


def Construct_Tree(optType, kappa, T, sigma, dt, r_0, c, K):
    # delta R
    dr = sigma * math.sqrt(3 * dt)
    # Band limit I
    I = math.floor(1 / (2 * dt * kappa)) + 1
    # Tree size pre-fixed
    steps = int(T / dt) + 1
    Tree = np.ndarray((steps,), object)
    # Loop
    for j in range(0, steps):
        # print('Number of node:',min(2*j+1,2*I+1))
        Tree_j = []
        Q_sum = 0
        for i in range(min(math.floor((2 * j + 1) / 2), I), max(math.floor((-2 * j - 1) / 2), -I - 1), -1):
            # i, j initialization
            TN = treeNode(i, j)
            # rij
            TN.rij = (1 - kappa * dt) * Tree[j - 1][i_to_arrayIndex(0, Tree[j - 1])].rij + TN.i * dr if j != 0 else r_0
            # m(i)
            TN.mi = i if abs(i) < I else i + 1 if i == -I else i - 1
            rho = -(kappa * TN.i * dt + (TN.mi - i)) * dr
            # Probability
            TN.pu = 0.5 * (((sigma ** 2 * dt + rho ** 2) / dr ** 2) + rho / dr)
            TN.pd = 0.5 * (((sigma ** 2 * dt + rho ** 2) / dr ** 2) - rho / dr)
            TN.pm = 1 - ((sigma ** 2 * dt + rho ** 2) / dr ** 2)
            # sign(i)
            i_sign = np.sign(TN.i) if TN.i != 0 else 1
            # parent i
            if TN.j == 0:
                TN.parent_i = None
            else:
                if abs(TN.i) > Tree[TN.j - 1][0].i:
                    TN.parent_i = [int(i_sign * (abs(TN.i) - 1))]
                elif abs(TN.i) == Tree[TN.j - 1][0].i:
                    if TN.j == 1:
                        TN.parent_i = [TN.i]
                    else:
                        TN.parent_i = sorted([TN.i, int(i_sign * (abs(TN.i) - 1))], reverse=True)
                else:
                    TN.parent_i = sorted([int(i_sign * (abs(TN.i) + 1)), TN.i, int(i_sign * (abs(TN.i) - 1))],
                                         reverse=True)
                    # band limit
                    if math.floor((2 * j + 1) / 2) > I:
                        if TN.i == I - 2:
                            TN.parent_i.append(I)
                        if TN.i == -I + 2:
                            TN.parent_i.append(-I)
                    TN.parent_i = sorted(TN.parent_i, reverse=True)
            # print('parent_i:',TN.parent_i)
            # child i
            if math.floor((2 * j + 1) / 2) >= I:
                TN.child_i = [TN.i, i_sign * (abs(TN.i) - 1), i_sign * (abs(TN.i) - 2)]
            else:
                TN.child_i = [TN.i + 1, TN.i, TN.i - 1]
            # print('child_i:',TN.child_i)
            # Arrow-Debreu Price
            if TN.j == 0:
                TN.Q = 1
            else:
                for pi in TN.parent_i:
                    parent_node = Tree[j - 1][i_to_arrayIndex(pi, Tree[TN.j - 1])]
                    probi = parent_node.pu if TN.i > pi else parent_node.pd if TN.i < pi else parent_node.pm
                    TN.Q += parent_node.Q * math.exp(-1 * parent_node.rij * dt) * probi
            # print('TN.Q:',TN.Q)
            Q_sum += TN.Q * math.exp(-1 * (TN.rij) * dt)
            # Node append at step j
            Tree_j.append(TN)
        # update interest rate
        # theta_j_1
        theta_j_1 = 1 / (kappa * (dt ** 2)) * math.log(Q_sum / (math.exp(-1 * r_0 * (j + 1) * dt)))
        for tn in Tree_j:
            tn.rij += kappa * theta_j_1 * dt
            # print('r'+str(tn.i)+str(tn.j)+':',tn.rij)
        # print('\n')
        # construct interest rate tree
        Tree[j] = Tree_j
    # Bond option pricing
    for j in range(steps - 1, -1, -1):
        # print('time:',j)
        for tn in Tree[j]:
            if j == steps - 1:
                tn.B = 1 + c * 0.5
                tn.optV = max(tn.B - K, 0)
            else:
                expV = 0
                C_sum = 0
                for ci in tn.child_i:
                    child_node = Tree[tn.j + 1][i_to_arrayIndex(ci, Tree[tn.j + 1])]
                    probi = tn.pu if ci > tn.i else tn.pd if ci < tn.i else tn.pm
                    expV += child_node.optV * math.exp(-1 * tn.rij * dt) * probi
                if optType[0] == 'E':
                    tn.optV = expV
                elif optType[0] == 'A':
                    for k in range(j, steps - 1):
                        C_sum += c * 0.5 / (
                                    (1 + tn.rij * 0.5) ** ((steps - 1 - k) * dt / 0.5)) if k * dt % 0.5 == 0 else 0
                        # print('C_sum:',C_sum)
                    C_sum = c * 0.5 + C_sum if (j != 0 and j * dt % 0.5 == 0) else C_sum
                    tn.B = C_sum + 1 / ((1 + tn.rij * 0.5) ** ((steps - 1 - j) * dt / 0.5))
                    # print('tn.B:',tn.B)
                    tn.optV = max(tn.B - K, expV, 0)
                    # print('r'+str(tn.i)+str(j)+'.optV:', tn.optV)
        # print('\n')
    return Tree
# dt = 0.5, European
IRTree_E_semi = Construct_Tree(optType='E', kappa=0.1, T=5, sigma=0.015, dt=0.5, r_0=0.05, c=0.05, K=1)
print('Option price with dt = 0.5: $',IRTree_E_semi[0][0].optV * 100)
# dt = 0.25, European
IRTree_E_quart = Construct_Tree(optType='European', kappa=0.1, T=5, sigma=0.015, dt=0.25, r_0=0.05, c=0.05, K=1)
print('Option price with dt = 0.25: $',IRTree_E_quart[0][0].optV * 100)
# dt = 0.5, American
IRTree_A_semi = Construct_Tree(optType='American', kappa=0.1, T=5, sigma=0.015, dt=0.5, r_0=0.05, c=0.05, K=1)
print('Option price with dt = 0.5: $',IRTree_A_semi[0][0].optV * 100)
# dt = 0.25, American
IRTree_A_quart = Construct_Tree(optType='A', kappa=0.1, T=5, sigma=0.015, dt=0.25, r_0=0.05, c=0.05, K=1)
print('Option price with dt = 0.25: $',IRTree_A_quart[0][0].optV * 100)
