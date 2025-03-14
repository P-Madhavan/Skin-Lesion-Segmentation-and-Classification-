import time

import numpy as np


def PROPOSED(Positions, fobj, Lb, Ub, Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    ub = Ub[1, :]
    lb = Lb[1, :]

    Fitness = np.zeros(N)
    Pbest_c = np.zeros(dim)

    V = Positions
    Pbest = Positions
    epochnumber = 2

    Convergence_curve = np.zeros(Max_iter)
    l = 0
    ct = time.time()

    while l < Max_iter:

        c1 = 2.05 * np.random.random()
        c2 = 2.05 * np.random.random()
        c3 = 2.05 * np.random.random()
        c4 = 2.05 * np.random.random()

        for i in range(N):

            Fitness[i] = fobj(Positions[i, :])
            Pbest[i, :] = lb + (ub - lb) * np.random.random(size=dim)

            if Fitness[i] < fobj(Pbest[i, :]):
                Pbest[i, :] = Positions[i, :]
            Pbest_c[i] = fobj(Pbest[i, :])

        xgbest = Pbest[0, :]
        cgbest = fobj(xgbest)

        for i in range(1, N):
            if fobj(Pbest[i, :]) < cgbest:
                xgbest = Pbest[i, :]
                cgbest = fobj(Pbest[i, :])

        galaxy_x = xgbest
        galaxy_c = cgbest

        count_f = 0

        for epoch in range(1, epochnumber):
            for r in range(Max_iter):
                for i in range(N):
                    r = ((Fitness[i]-np.min(Fitness))/(np.max(Fitness)-np.min(Fitness)))
                    r2 = -1 + 2 * np.random.random()
                    v = c1 * r * Pbest[i, :] - Positions[i, :] + c2 * r2 * xgbest - Positions[i, :]
                    V[i, :] = (1 - r / (Max_iter + 1)) * V[i, :] + v
                    for j in range(dim):
                        V[i, j] = np.max([V[i, j], Lb[i, j]])
                        V[i, j] = np.min([V[i, j], Ub[i, j]])
                    Positions[i, :] = Positions[i, :] + V[i, :]
                    for j in range(dim):
                        Positions[i, j] = np.max([Positions[i, j], Lb[i, j]])
                        Positions[i, j] = np.min([Positions[i, j], Ub[i, j]])
                    Fitness[i] = fobj(Positions[i, :])
                    count_f += 1
                    if Fitness[i] < Pbest_c[i]:
                        Pbest[i, :] = Positions[i, :]
                        Pbest_c[i] = Fitness[i]
                        if Pbest_c[i] < cgbest:
                            xgbest = Pbest[i, :]
                            cgbest = Pbest_c[i]

            for t in range(Max_iter):
                for i in range(N):
                    r = ((Fitness[i]-np.min(Fitness))/(np.max(Fitness)-np.min(Fitness)))
                    r2 = -1 + 2 * np.random.random()
                    v = c3 * r * Pbest[i, :] - Positions[i, :] + c4 * r2 * xgbest - Positions[i, :]
                    V[i, :] = (1 - t / (Max_iter + 1)) * V[i, :] + v
                    for j in range(dim):
                        V[i, j] = np.max([V[i, j], Lb[i, j]])
                        V[i, j] = np.min([V[i, j], Ub[i, j]])
                    Positions[i, :] = Positions[i, :] + V[i, :]
                    for j in range(dim):
                        Positions[i, j] = np.max([Positions[i, j], Lb[i, j]])
                        Positions[i, j] = np.min([Positions[i, j], Ub[i, j]])
                    Fitness[i] = fobj(Positions[i, :])
                    count_f += 1
                    if Fitness[i] < Pbest_c[i]:
                        Pbest[i, :] = Positions[i, :]
                        Pbest_c[i] = Fitness[i]
                        if Pbest_c[i] < cgbest:
                            xgbest = Pbest[i, :]
                            cgbest = Pbest_c[i]

        Convergence_curve[l] = cgbest
        l = l + 1

    ct = time.time() - ct
    return cgbest, Convergence_curve, xgbest, ct
