import numpy as np
import scipy.linalg as lin
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
import matplotlib.pyplot as plt
import time



def constructHPBC(N,gamma,epsilons):
    """diagonalization"""
    def H(i,j):
        if i==j:
            return epsilons[i]
        if j==i+1 or j == i-1 or (i == 0 and j==N-1) or (i == N-1 and j==0):
            return gamma
        else:
            return 0
    matrix = np.array([[H(j,i) for i in range(N)] for j in range(N)])
    return lin.eigh(matrix)

def IPRContour(gamma,N,filename):
    """Contour plot of IPR ( randomness) for many energies"""
    deltas = [0.05*i for i in range(51)]
    energies = []
    IPRs = []
    epsilons = []
    for delta in deltas:
        print(delta)
        epsilon = 0.5*delta*(2*np.random.rand(N)-1)
        res = constructHPBC(N,gamma,epsilon)
        energy, indices = np.unique(res[0],return_index=True)
        IPR = [1/np.sum(res[1][:,i]**4) for i in range(N) if i in indices]
        IPRs = IPRs + IPR
        energies   = energies + energy.tolist()
        epsilons = epsilons + [delta for i in range(len(IPR))]
    
    cnt = plt.tricontourf(energies,epsilons,IPRs,levels=np.linspace(np.amin(IPRs),np.amax(IPRs),50))
    plt.colorbar(cnt)
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\Delta $")
    plt.title(r"Tight Binding: IPR v odvisnosti od $E$ in $\Delta$, $N={}, \gamma={}$".format(N,gamma))
    #plt.savefig(filename)
    
    
IPRContour(1,1000,"lol")

def densityOfStates(gamma,N,filename,plot="both"):
    """Calculates the density of states"""
    deltas = [0.1*i for i in range(20)]
    
    def cauchy(x,x0):
        HWHM = 0.01
        return 1/np.pi * HWHM / ((x-x0)**2 +HWHM**2) 
    def density(x,states):
        suma=0
        for r in states:
            suma += cauchy(x,r)
        return suma


    if plot=="both":
        #comparison of two DOS methods
        res = constructHPBC(N,gamma,np.zeros(N))[0]
        fig, [ax1,ax2] = plt.subplots(1,2,sharey=True)
        ax1.set_xlabel(r"$E$")
        ax2.set_xlabel(r"$E$")
        rang = np.linspace(np.amin(res),np.amax(res),100)
        difference = rang[1]-rang[0]
        numstates = [np.logical_and(res>rang[i],res<rang[i+1]).sum() for i in range(99)]
        res1 = np.abs(np.gradient(numstates,difference))
        ax1.plot(rang[:-1],2*N*res1/(np.sum(res1)*difference))
        es = np.linspace(np.amin(res),np.amax(res),100)
        res2 = np.array([density(r,res) for r in es])
        ax2.plot(es,2*N*res2/(np.sum(res2)*(es[1]-es[0])))
        ax1.set_title(r"$g(E)$ z odvodom")
        ax2.set_title(r"$g(E)$ z delta funkcijami")
        plt.savefig(filename)
        
    elif plot=="subplots":
        #dos on several seperate subplots
        deltas = [0,0.5,1,1.5,2,2.5,3,3.5,4]
        fig, axi = plt.subplots(3,3,sharey=True)
        axi = axi.flatten()
        for i in range(9):
            print(i)
            delta = deltas[i]
            epsilons = 0.5*delta*(2*np.random.rand(N)-1)
            res = constructHPBC(N,gamma,epsilons)[0]
            es = np.linspace(np.amin(res),np.amax(res),100)
            difference = es[1]-es[0]
            result = np.array([density(e,res) for e in es])
            axi[i].plot(es,2*N*result/(np.sum(result)*difference))
            axi[i].set_title("$\Delta = {}$".format(deltas[i]))
        plt.tight_layout()
        plt.savefig(filename)
    
        
    elif plot=="oneplot":
        #many graphs on one subplot
        deltas = [0.1*i for i in range(11)]
        cmap = plt.get_cmap("brg")
        colors = np.linspace(0,1,11)
        lower = 0
        fig,ax=plt.subplots()
        for i in range(11):
            print(i)
            delta = deltas[i]
            epsilons = 0.5*delta*(2*np.random.rand(N)-1)
            res = constructHPBC(N,gamma,epsilons)[0]
            es = np.linspace(np.amin(res),np.amax(res),100)
            difference = es[1]-es[0]
            result = np.array([density(e,res) for e in es])
            result = 2*N*result/(np.sum(result)*difference)
            lower+= np.amax(result)+1
            ax.plot(es,result+np.ones(100)*lower,color=cmap(colors[i]),label=r"$\Delta={}$".format(round(delta,2)))        
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(r"$E$")
        ax.legend(bbox_to_anchor=(1,0.5),loc="center")
        ax.set_title(r"Gostota stanj za model tesne vezi, $N={}, \gamma={}$".format(N,gamma))
        plt.savefig(filename)

def dispersion(N,gamma):
    """ Dispersion """
    epsilons=np.zeros(N)
    ks = np.linspace(-np.pi,np.pi,N)
    res = constructHPBC(N,gamma,epsilons)[0]
    energies = np.concatenate((res[1:-1][::2][::-1],[res[0]],res[1:-1][1::2],[res[-1]]))
    plt.plot(ks,energies)
    

    
def naiveLocalization(N,gamma,filename):
    """ Naive way of determining wavefunction size"""
    deltas = [0.1*i for i in range(1,101)]
    sizes = []
    for delta in deltas:
        print(delta)
        size = 0
        for j in range(100):
            epsilons = 0.5*delta*(np.random.rand(N)-1)
            res = constructHPBC(N,gamma,epsilons)[1][:,0]
            size+=sum([1 if abs(i)>0.01 else 0 for i in rez])
        sizes.append(size/100)
    plt.plot(deltas,sizes)
    plt.xlabel(r"$\Delta \epsilon$")
    plt.ylabel(r"$l$")
    plt.title(r"Širina valovne funkcije $l$ v odv. od $\Delta \epsilon$ povprečeno po 100 realizacijah")
    plt.savefig(filename)
    
    
    
    
    
    
    
    
    
    
#    
#IPRContour(1,1000,"IPRN10002.pdf")
#densityOfStates(1,1000,"PrimerjavaDOS.pdf","both")
#densityOfStates(1,1000,"GostoteStanj.pdf","subplots")
#densityOfStates(1,1000,"GostoteStanj2.pdf","oneplot")
#dispersion(1000,1)
#naiveLocalization(100,1,"Velikosti.pdf")









def skonstruirajHOBC(N,gamma,epsiloni):
    """Open boundary conditions, Irrelevant"""
    obdiagonalci = np.ones(N-1)*gamma
    return lin.eigh_tridiagonal(epsiloni,obdiagonalci,check_finite=False)

def skonstruirajHPBCSparse(N,gamma,epsiloni):
    """PBC with sparse matrices. Irrelevant"""
    data = np.concatenate((np.ones(2*N)*gamma,epsiloni))
    i = np.concatenate((np.array([0,N-1]),np.arange(N-1),np.arange(1,N),np.arange(N)))
    j = np.concatenate((np.array([N-1,0]),np.arange(1,N),np.arange(N-1),np.arange(N)))
    matrika = sparse.coo_matrix((data,(i,j)),shape=(N,N))
    return slin.eigsh(matrika,N-1)



def speedComparison():
    """Sparse vs tridiagonal. Irrelevant"""
    #primerjava hitrosti sparse vs tridiagonal
    tridiagTimes  = []
    normalTimes = []
    gamma = 1
    Ns = [100*i for i in range(1,21)]
    for N in Ns:
        print(N)
        epsiloni = np.zeros(N)
        start = time.time()
        skonstruirajHOBC(N,gamma,epsiloni)
        tridiagTimes.append(time.time()-start)
        start = time.time()
        skonstruirajHPBC(N,gamma,epsiloni)
        normalTimes.append(time.time()-start)
        
    plt.plot(Ns, tridiagTimes, ls=":",marker=".",label = "Tridiagonal")
    plt.plot(Ns, normalTimes, ls=":",marker=".",label="Navadno")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("t[s]")
    plt.title("Primerjava časovne zahtevnosti dveh metod za diagonalizacijo")
    plt.grid()
    plt.savefig("Primerja.pdf")