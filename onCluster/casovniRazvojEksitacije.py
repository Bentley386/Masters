"""
To run on cluster. Excitations during the time evolution.
"""
import numpy as np
import scipy.linalg as lin
import sys
import mpmath as mp 
import scipy.sparse as sparse
import scipy.sparse.linalg as lins

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


#Total number of excitations for different quench times

seed = int(sys.argv[1])
print(seed)
np.random.seed(seed)
N=1000
omega1 = np.random.rand(N)-0.5
omega2 = np.random.rand(N)-0.5
tau=1
Winitial = 3.5
Wfinal = 4.5   
res = []
for T in np.linspace(10,1000,50):
    print(T)
    steps = int(T/tau)+1
    ms = np.zeros(steps)
    Ws = np.linspace(Winitial,Wfinal,steps)[::-1]
    vs = np.ones(N)*ms[0] + Ws[0]*omega2
    ws = np.ones(N)+0.5*Ws[0]*omega1
    psis = constructHPBC(N,vs,ws,np.zeros(N))[1][:,:int(N/2)]
    
    for i in range(1,steps):
        vs = np.ones(N)*ms[i] + Ws[i]*omega2
        ws = np.ones(N)+0.5*Ws[i]*omega1
        b = np.matmul(timeEvolutionMatrix(N,0.5*tau,vs,ws).todense(),psis)
        psis = lins.spsolve(timeEvolutionMatrix(N,-0.5*tau,vs,ws).tocsc(),b)

    if T==10:
        final = constructHPBC(N,vs,ws,np.zeros(N))[1][:,int(N/2):]
        P0 = np.zeros((N,N),dtype=np.complex128) 
        for i in range(int(N/2)):
            P0 = P0 + np.tensordot(np.conjugate(final[:,i]),final[:,i],0)
    P = np.zeros((N,N),dtype=np.complex128)
    for j in range(int(N/2)):
        P = P + np.tensordot(np.conjugate(psis[:,j]),psis[:,j],0)
        
    res.append(np.trace(np.matmul(P,P0)))

with open("/{}.txt".format(seed),"w") as f:
    f.write(" ".join(list(map(str,res))))
