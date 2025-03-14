import numpy as np
import random as rn
import math
import time


def DHOA(Positions,fobj,VRmin,VRmax,Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    Wind_angle = 2 * math.pi * rn.random()
    pos_angle = Wind_angle + math.pi
    Leader_pos = np.zeros((dim, 1))
    Leader_score = float('inf')

    sc_pos = np.zeros((dim, 1))
    sc_score = float('inf')

    Convergence_curve = np.zeros((Max_iter, dim))

    t = 0
    ct = time.time()
    while t < Max_iter:
        for i in range(N):
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])

            #  Update the leader
            if fitness < Leader_score:
                Leader_score = fitness  # Update alpha
                Leader_pos = Positions[i, :]

            if  Leader_score < fitness < sc_score:
                sc_score = fitness
                sc_pos = Positions[i, :]

        # Update the Position of search agents
        for i in range(N):
            r = (math.pi / 8) * rn.random()
            v = Wind_angle - r
            A = pos_angle + v
            m1 = 0
            m2 = 2
            p = (m2 - m1)* rn.random() + m1
            a = -1; b = 1
            for j in range(dim):
                r1 = a + (b - a)* rn.random()   # r1 is a random number in [-1,1]
                r2 = rn.random()  # r2 is a random number in [0,1]
                if p < 1:
                    A1 = (1 / 4) * math.log(t + (1 / Max_iter)) * r1
                    C1 = 2 * r2
                    if abs(C1) >= 1:
                        alp = abs((C1 * Leader_pos[j]) - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A1 * p * alp
                    else:
                        alp = abs((C1 * sc_pos[j]) - Positions[i, j])
                        Positions[i, j] = sc_pos[j] - A1 * p * alp
                else:
                    alp = abs((math.cos(A) * Leader_pos[j]) - Positions[i, j])
                    Positions[i, j] = Leader_pos[j] - r1 * p * alp
        Convergence_curve[t,:] = Leader_pos
        t = t + 1
    Leader_score = Convergence_curve[Max_iter - 1]
    ct = time.time() - ct

    return Leader_score, Convergence_curve, Leader_pos, ct
