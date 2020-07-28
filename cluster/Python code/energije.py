"""
To run on cluster. Energies during quench.
"""
import numpy as np
import scipy.linalg as lin
import sys
import mpmath as mp 
import scipy.sparse as sparse
import matplotlib.pyplot as plt

mp.mp.dps = 20
#factor = 1e100
factor = 1


def constructHPBC(N,v,w,epsilons):
    """diagonalization"""
    if np.isscalar(v):
        v = np.ones(N)*v
        w = np.ones(N)*w
        
    def H(i,j):
        if i==j:
            return epsilons[i]
        if j==i+1:
            if i%2==0:
                #return v[i]*1j
                return 0
            else:
                return 0
                return w[i]
        if j == i-1:
            if i%2==1:
                return v[i]*(-1j)
            else:
                return w[i]
        if (i == 0 and j==N-1):
            return 0
            return w[0]
        if i==N-1 and j==0:
            return w[0]
        else:
            return 0
        
    matrix = np.array([[H(j,i) for i in range(N)] for j in range(N)],dtype=np.complex128)
    return lin.eigh((matrix*factor).astype(np.complex128))


def timeEvolutionMatrix(N,tau,vs,ws):
    faktor = -1j*tau*0.5
    diagonal = np.ones(N,dtype=np.complex128)
    subdiagonal = np.ones(N,dtype=np.complex128)
    subdiagonal[::2] = vs[1::2]*1j
    subdiagonal[1::2] = np.append(ws[2::2],0)
    edge = np.ones(N)*ws[0]
    data = np.vstack([diagonal,faktor*np.conjugate(subdiagonal),faktor*np.roll(subdiagonal,1),faktor*edge,faktor*edge])
    offsets = [0,-1,1,N-1,-N+1]
    return sparse.dia_matrix((data,offsets),shape=(N,N))

#All energies on a path across phase transition
    


#seed = int(sys.argv[1])
seed = 8166247
np.random.seed(seed)
N=1000
omega1 = np.random.rand(N)-0.5
omega2 = np.random.rand(N)-0.5
Winitial = 3
Wfinal = 5           
Ws = np.linspace(Winitial,Wfinal,200)
res = []
for W in Ws:
    vs = W*omega2
    ws = np.ones(N)+0.5*W*omega1
    energies = constructHPBC(N,vs,ws,np.zeros(N))[0]
    res.append(energies)

#SAVE FILE    
#with open("rezultati/{}.txt".format(seed),"w") as f:
#    f.write(" ".join(list(map(str,res))))


res = np.array(res)
for i in range(10):
    plt.plot(Ws,res[:,int(N/2)+i])
plt.yscale("log")