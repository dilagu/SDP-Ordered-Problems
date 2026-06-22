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

    Z = [M.variable(f"Z{j}", Domain.inPSDCone(3*n)) for j in range(p)]

    d = M.variable("d", [n,p], Domain.greaterThan(0))
    y = M.variable("y", [p,dim], Domain.unbounded())

    M.objective(ObjectiveSense.Minimize, 0.5*trace(sum(B)))

    M.constraint(sum(P) @ Matrix.ones(n,1) == Matrix.ones(n,1))

    for j in range(p):
        M.constraint(trace(P[j]) == 1)

        M.constraint(Z[j] == Expr.stack([
            [Expr.constTerm(Matrix.eye(n)), P[j], D[j]],
            [P[j], P[j], B[j].T],
            [D[j], B[j], C[j]],
        ]))
        M.constraint(Z[j] >= 0)

        for i in range(n):
            dd = Expr.vstack([d[i,j]] + [y[j,k] - demands[i][k] for k in range(dim)])
            M.constraint(dd, Domain.inQCone())

            for k in range(i+1):
                M.constraint(D[j][i,k] == d[i,j] + d[k,j])

        if ineq["I<P sdp bound"]:
            M.constraint(Matrix.eye(n) - sum(P), Domain.inPSDCone(n))
        
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

        if ineq["boros hammer"]:
            for _ in range(10):
                w = np.random.randint(-10, 10+1, n+1)
                M.constraint(w[n]*(w[n]-1) + sum([w[i]*(w[i] + 2*w[n] - 1)*P[j][i,i] for i in range(n)]) + 2*sum([w[i]*w[k]*P[j][i,k] for k in range(i) for i in range(n)]) >= 0)

        if ineq["distance projection linking cuts"]:
            for i in range(n):
                for k in range(i):
                    deltaik = np.linalg.norm(np.array(demands[i]) - np.array(demands[k]))
                    M.constraint(D[j][i,k] >= deltaik)
                    M.constraint(B[j][i,k] >= deltaik * P[j][i,k])

        if ineq["distance square diameter bounds"]:
            M.constraint(C[j] <= Matrix.dense(n,n,n*diam2*4))
            
        if ineq["cluster symmetry break"]:
            for i in range(p-1):
                for j in range(i+1,p):
                    M.constraint(P[j][i,i] == 0)
        
    
    if ineq["rlt assignment consistency"]:
        for i in range(n):
            for k in range(n):
                M.constraint(sum([P[j][i,i] + P[j][k,k] - 2 * P[j][i,k] for j in range(p)]) >= 0)
    
    M.solve()

    currentP = [np.reshape(P[j].level(), (n,n)) for j in range(p)]
    currentB = [np.reshape(B[j].level(), (n,n)) for j in range(p)]
    currentC = [np.reshape(C[j].level(), (n,n)) for j in range(p)]
    currentD = [np.reshape(D[j].level(), (n,n)) for j in range(p)]
    currentZ = [np.reshape(Z[j].level(), (3*n,3*n)) for j in range(p)]
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
    "I<P sdp bound" : False,
    "diagonal bounds" : False,
    "off-diagonal bounds" : False,
    "2x2 minors" : False,
    "triangle inequalities" : False,
    "distance sym and nn" : False,
    "pairwise distance lb" : False,
    "eigencuts" : False,
    "sparse eigencuts" : False,
    "sparse dnn eigencuts" : False,
    "boros hammer" : False,
    "clique inequalities" : False,
    "odd-cycle inequalities" : False,
    "socrlt perspective cuts" : False,
    "rlt assignment consistency" : False,
    "distance projection linking cuts" : False,
    "geometric conflict cuts" : False,
    "distance square diameter bounds" : False,
    "cluster symmetry break" : False
}

for k in inequalities.keys():
    inequalities[k] = True

weber(demands, p, inequalities)