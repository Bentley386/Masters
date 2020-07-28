"""
To run on cluster. Transitions by state during quench.
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
            seedi = "\n".join(list(map(str,list(map(int,np.random.rand(100)*10**12)))))
            f.write(seedi)

def excToState(final,psis):
	prob=0
	numStates = len(psis[0])
	for state in range(numStates):
		prob+= np.abs(np.vdot(final,psis[:,state]))**2
	return prob

def excToAllStates(interested,others):
    occupation = []
    for i in range(1000):
        occupation.append(np.abs(np.vdot(interested,others[:,i]))**2)
    return occupation
if 1:
    #casovni razvoj
    seed = int(sys.argv[1])
    #seed = 8166247
    np.random.seed(seed)
    N=1000
    omega1 = np.random.rand(N)-0.5
    omega2 = np.random.rand(N)-0.5
    tau=1
    T=6000
    results = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
    energies = []
    
    interestedWs=list(np.arange(3*10+1,5*10+1)/10)
    koraki = int(T/tau)+1
    mji = np.zeros(koraki)
    Wji = np.linspace(3,5,koraki)
    vji = np.ones(N)*mji[0] + Wji[0]*omega2
    wji = np.ones(N)+0.5*Wji[0]*omega1
#    energije=skonstruirajHPBC(N,vji,wji,np.zeros(N))[0][:int(N/2)]
#    for T in results:
#        print(energije[-T])
#    č
    psiji = skonstruirajHPBC(N,vji,wji,np.zeros(N))[1][:,:int(N/2)]
        
    for i in range(1,koraki):
        vji = np.ones(N)*mji[i] + Wji[i]*omega2
        wji = np.ones(N)+0.5*Wji[i]*omega1
        b = np.matmul(matrikaZaCasovniRazvoj(N,0.5*tau,vji,wji).todense(),psiji)
        psiji = lins.spsolve(matrikaZaCasovniRazvoj(N,-0.5*tau,vji,wji).tocsc(),b)
            
        if np.argmin(np.abs(Wji-interestedWs[0])) == i:
            interestedWs.pop(0)
            koncne = skonstruirajHPBC(N,vji,wji,np.zeros(N))
            koncneFunkcije = koncne[1]
            koncneEnergije = koncne[0]
            energies.append(koncneEnergije)
            for T in results:
                results[T].append(excToAllStates(psiji[:,-T],koncneFunkcije))

                
with open("poStanjih/energije/{}.txt".format(seed),"w") as f:
    f.write(" ".join(list(map(str,energies))))
for T in results:
    with open("poStanjih/eksitacije/{}/{}.txt".format(T,seed),"w") as f:
        f.write(" ".join(list(map(str,results[T]))))
