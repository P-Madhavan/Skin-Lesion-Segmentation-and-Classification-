from time import time
import numpy as np
import random as rn


def TSA(Positions, fobj, LB, UB, Max_iter):
    Pos = Positions
    N, dim = Positions.shape[0], Positions.shape[1]
    ub = UB[0, :]
    lb = LB[0, :]
    Fitness = np.zeros(N)
    Position = np.zeros(dim)
    Score = np.Inf
    Convergence = np.zeros(Max_iter)
    ct = time()
    for t in range(Max_iter):
        for i in range(N):
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = Positions[i, :] * (np.bitwise_not(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb

            fitness = fobj(Positions[i, :])

            if fitness < Score:
                Score = fitness
                Position = Positions[i, :]

        xmin = 1
        xmax = 4
        xr = xmin + (xmax - xmin) * rn.random()
        xr = np.round(xr)

        for i in range(N):
            for j in range(dim):
                A1 = ((rn.random() + rn.random()) - (2 * rn.random())) / xr
                c2 = rn.random()
                if i == 1:
                    c3 = rn.random()
                    if c3 >= 0:
                        d_pos = abs(Position[j] - (c2 * Positions[i, j]))
                        Positions[i, j] = Position[j] + (A1 * d_pos)
                    else:
                        d_pos = abs(Position[j] - (c2 * Positions[i, j]))
                        Positions[i, j] = Position[j] - (A1 * d_pos)
                else:
                    c3 = rn.random()
                    if c3 >= 0:
                        d_pos = abs(Position[j] - (c2 * Positions[i, j]))
                        Pos[i, j] = Position[j] + (A1 * d_pos)
                    else:
                        Pos[i, j] = Position[j] - (A1 * d_pos)
                    Positions[i, j] = (Pos[i, j] + Positions[i - 1, j]) / 2
        Convergence[t] = Score
    ct = time() - ct
    return Score, Convergence, Position, ct