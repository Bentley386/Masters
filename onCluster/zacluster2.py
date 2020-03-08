
import numpy as np
import scipy.linalg as lin
import sys
import mpmath as mp 
import scipy.sparse as sparse

mp.mp.dps = 20
#factor = 1e100
factor = 1


def skonstruirajHPBC(N,v,w,epsiloni):
    """periodicni, navadna diagonalizacija"""
    if np.isscalar(v): #dejansko ne rabimo N vjev, en bo zmeraj ostal neuporabljen
        v = np.ones(N)*v
        w = np.ones(N)*w
        
    def H(i,j):
        if i==j:
            return epsiloni[i]
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
        
    matrika = np.array([[H(j,i) for i in range(N)] for j in range(N)],dtype=np.complex128)
    #return matrika
    #matrika = sympy.Matrix([[H(j,i) for i in range(N)] for j in range(N)])
    #return matrika.eigenvals().keys()
    #return mp.eigsy(mp.matrix(matrika),eigvals_only=True)
    return lin.eigh((matrika*factor).astype(np.complex128))


def matrikaZaCasovniRazvoj(N,tau,vji,wji):
    faktor = -1j*tau*0.5
    diagonala = np.ones(N,dtype=np.complex128)
    poddiagonala = np.ones(N,dtype=np.complex128)
    poddiagonala[::2] = vji[1::2]*1j
    poddiagonala[1::2] = np.append(wji[2::2],0)
    robna = np.ones(N)*wji[0]
    data = np.vstack([diagonala,faktor*np.conjugate(poddiagonala),faktor*np.roll(poddiagonala,1),faktor*robna,faktor*robna])
    offsets = [0,-1,1,N-1,-N+1]
    return sparse.dia_matrix((data,offsets),shape=(N,N))

if 0:
    with open("seedi.txt","w") as f:
            seedi = "\n".join(list(map(str,list(map(int,np.random.rand(100)*10**8)))))
            f.write(seedi)

if 1:
    #Energije cez fazni prehod
    seed = int(sys.argv[1])
    np.random.seed(seed)
    N=1000
    omega1 = np.random.rand(N)-0.5
    omega2 = np.random.rand(N)-0.5
    Winitial = 3.5  #fazni prehod okoli 3.9
    Wfinal = 4.5   # W(t) = Winitial + (Wfinal-Winitial)/T * t        
    Wji = np.linspace(Winitial,Wfinal,100)
    rez = []
    for W in Wji:
        vji = W*omega2
        wji = np.ones(N)+0.5*W*omega1
        energije = skonstruirajHPBC(N,vji,wji,np.zeros(N))[0]
        rez.append(energije)
    with open("rezultati.txt","w") as f:
        f.write(" ".join(list(map(str,rez))))