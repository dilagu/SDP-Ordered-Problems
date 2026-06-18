import mosek
from mosek.fusion import *
import mosek.fusion.pythonic
import numpy as np
from math import sqrt
import sys

np.set_printoptions(precision=5, suppress=True)

def trace(X):
    return Expr.dot(Matrix.eye(X.shape[0]),X)

def weber(demands, p, ineq):
    n = len(demands)
    dim = len(demands[0])

    diam2 = 0
    for i in range(n):
        for k in range(i):
            temp = sum([(demands[i][l] - demands[k][l])**2 for l in range(dim)])
            if temp > diam2:
                diam2 = temp
    
    M = Model()
    M.setSolverParam("log", 20)
    M.setLogHandler(sys.stdout)

    P = [M.variable(f"P{j}", Domain.inPSDCone(n)) for j in range(p)]
    D = [M.variable(f"D{j}", [n,n], Domain.greaterThan(0)) for j in range(p)]
    B = [M.variable(f"B{j}", [n,n], Domain.greaterThan(0)) for j in range(p)]
    C = [M.variable(f"C{j}", Domain.inPSDCone(n)) for j in range(p)]

    d = M.variable("d", [n,p], Domain.greaterThan(0))
    y = M.variable("y", [p,dim], Domain.unbounded())

    M.objective(ObjectiveSense.Minimize, 0.5*trace(sum(B)))

    M.constraint(sum(P) @ Matrix.ones(n,1) == Matrix.ones(n,1))

    for j in range(p):
        M.constraint(trace(P[j]) == 1)

        Zj = Expr.stack([
            [Expr.constTerm(Matrix.eye(n)), P[j], D[j]],
            [P[j], P[j], B[j].T],
            [D[j], B[j], C[j]],
        ])
        M.constraint(Zj, Domain.inPSDCone(3*n))
        M.constraint(Zj >= 0)

        for i in range(n):
            dd = Expr.vstack([d[i,j]] + [y[j,k] - demands[i][k] for k in range(dim)])
            M.constraint(dd, Domain.inQCone())

            for k in range(i+1):
                M.constraint(D[j][i,k] == d[i,j] + d[k,j])
        
        if ineq["I<P sdp bound"]:
            M.constraint(Matrix.eye(n) - sum(P), Domain.inPSDCone(n))

        if ineq["distance square diameter bounds"]:
            M.constraint(C[j] <= Matrix.dense(n,n,n*diam2*4))
        
        if ineq["diagonal bounds"]:
            for i in range(n):
                M.constraint(P[j][i,i] <= 1)
        
        if ineq["off-diagonal bounds"]:
            for i in range(n):
                for k in range(i):
                    M.constraint(P[j][i,k] <= P[j][i,i])
                    M.constraint(P[j][i,k] <= P[j][k,k])
        
        if ineq["2x2 minors"]:
            for i in range(n):
                for k in range(i):
                    M.constraint(Expr.stack([
                        [P[j][i,i], P[j][i,k]],
                        [P[j][k,i], P[j][k,k]]
                    ]), Domain.inPSDCone())
        
        
        if ineq["nxn minors"]:
            for i in range(n):
                for k in range(i):
                    M.constraint(Expr.stack([
                        [P[j][i,i], P[j][i,k]],
                        [P[j][k,i], P[j][k,k]]
                    ]), Domain.inPSDCone())
        
        if ineq["triangle inequalities"]:
            for i in range(n):
                for k in range(i):
                    for l in range(i):
                        M.constraint(P[j][i,k] >= P[j][i,l] + P[j][k,l] - P[j][l,l])
                        M.constraint(P[j][i,l] >= P[j][i,k] + P[j][k,l] - P[j][k,k])
                        M.constraint(P[j][k,l] >= P[j][i,k] + P[j][i,l] - P[j][i,i])
                        
                        M.constraint(P[j][i,k] + P[j][i,l] + P[j][k,l] >= P[j][i,i] + P[j][k,k] + P[j][l,l] - 1)
        
        if ineq["distance sym and nn"]:
            M.constraint(D[j].T == D[j])
            M.constraint(D[j] >= 0)
        
        if ineq["pairwise distance lb"]:
            for i in range(n):
                for k in range(i):
                    M.constraint(D[j][i,k] >= sqrt(sum([(demands[i][r] - demands[k][r])**2 for r in range(dim)]))) # unweighted
    
    M.solve()

    currentP = [np.reshape(P[j].level(), (n,n)) for j in range(p)]
    currentB = [np.reshape(B[j].level(), (n,n)) for j in range(p)]
    currentC = [np.reshape(C[j].level(), (n,n)) for j in range(p)]
    currentD = [np.reshape(D[j].level(), (n,n)) for j in range(p)]
    currenty = y.level()

    score = 0

    for j in range(p):
        print(currentP[j])
        eigvals = np.linalg.eigh(currentP[j]).eigenvalues
        score += eigvals[-1]/p*100

        print(eigvals)
        #print(currentB[j])
        #print(currentC[j])
        print()
    print(currenty)
    
    print()

    print(f"SCORE = {score}/100") # 100 = EXACTO
    print()


demands = [[0,0],[1,0],[0,1]]
p = 2

inequalities = {
    "I<P sdp bound" : True,
    "diagonal bounds" : True,
    "off-diagonal bounds" : True,
    "2x2 minors" : True,
    "nxn minors" : True,
    "triangle inequalities" : True,
    "distance sym and nn" : True,
    "pairwise distance lb" : True,
    "eigencuts" : True,
    "sparse eigencuts" : True,
    "sparse dnn eigencuts" : True,
    "boros hammer" : True,
    "clique inequalities" : True,
    "odd-cycle inequalities" : True,
    "socrlt perspective cuts" : True,
    "rlt assignment consistency" : True,
    "distance projection linking cuts" : True,
    "geometric conflict cuts" : True,
    "distance square diameter bounds" : True
}

weber(demands, p, inequalities)