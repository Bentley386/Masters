
import numpy as np
import scipy.linalg as lin
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpmath as mp 
from scipy.special import binom
import scipy.sparse as sparse
import scipy.sparse.linalg as lins
import numba
import time
from numba import jit

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

def cTab(N):
     """Calculates the finite differential coefficients, first argument
 is the length of the sample"""
     A = mp.matrix(N + 1)
     for i in range(N + 1):
         for j in range(N + 1):
             A[i,j] = mp.mpmathify(- N//2 + j) ** mp.mpmathify(i)
     b = mp.matrix(N + 1, 1)
     b[1] = mp.mpmathify(500) / (2*mp.pi)
     c = mp.lu_solve(A, b)
     return np.array(c.tolist(), float)[:, 0]

def csToDict(cs):
        """assumaming odd number of coefficients"""
        N = int(0.5*(len(cs)-1))
        keys = [i for i in range(-N,N+1)]
        return {keys[i] : cs[i] for i in range(2*N+1)}





def windingNumber(N,n,vectors,cs,SMinus,SPlus,firstexp,secondexp):
    PMinus = np.zeros((N,N),dtype=np.complex128)
    for i in range(int(N/2)):
        PMinus += np.tensordot(np.conjugate(vectors[:,i]),vectors[:,i],0)
    Q = np.identity(N,np.complex128)-2*PMinus
    QMinusPlus = np.matmul(SMinus,np.matmul(Q,SPlus))
    QPlusMinus = np.matmul(SPlus,np.matmul(Q,SMinus))    
    commutator = np.zeros((N,N),dtype=np.complex128)
    for c in range(n+1):
        commutator += cs[c]*np.matmul(firstexp[c],np.matmul(QPlusMinus,secondexp[c]))
    return 1/(0.5*N)*np.trace(np.matmul(QMinusPlus,1j*commutator))


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

def lengthFormula(m,W):
    """localization length as in article"""
    W2 = W
    W1 = 0.5*W
    numerator = np.abs(2+W1)**(1/W1 + 0.5)*np.abs(2*m - W2)**(m/W2 - 0.5)
    denominator = np.abs(2-W1)**(1/W1-0.5)*np.abs(2*m + W2)**(m/W2 + 0.5)
    return np.abs(np.log(numerator/denominator))


def lengthEmpiric(vs,ws):
    """loc. length by averaging"""
    ms = vs
    ts = ws  #notation in article
    n = len(ms)
    lamb = np.sum(np.log(np.abs(ts))-np.log(np.abs(ms)))/n
    return np.abs(lamb)


def localization(N,plot,filename):
    """plots related to localization length"""
    if plot=="ForMasters":
        #final graph to go in the masters
        m=0
        Ws = np.linspace(3,5,200)
        plt.grid(True)
        plt.plot(Ws,[lengthFormula(0,W) for W in Ws],label="Formula")
        for N in [1000,2000,3000]:
            result = np.zeros(200)
            for k in range(1):
                omega1 = np.random.rand(N)-0.5
                omega2 = np.random.rand(N)-0.5
                for i in range(200):
                    W = Ws[i]
                    vs = W*omega2
                    ws = np.ones(N)+0.5*W*omega1
                    result[i] += lengthEmpiric(vs,ws)
            plt.plot(Ws,result,label=r"$N={}$".format(N))
        plt.xlabel(r"$W$")
        plt.ylabel(r"$\Lambda^{-1}$")
        plt.title("Primerjava asimptotske formule za lokalizacijsko dolžino s simulacijami")
        plt.legend()
        plt.savefig(filename)
            
    if plot=="IPR":
        #IPR Histogram
        m=0
        W=4
        W2 = W
        W1 = 0.5*W
        IPRs = []
        for i in range(1000):
            if i%100==0:
                print(i)
            omega1 = np.random.rand(N)-0.5
            omega2 = np.random.rand(N)-0.5
            vs = np.ones(N)*m + W2*omega2
            ws = np.ones(N)+W1*omega1
            vectors = constructHPBC(N,vs,ws,np.zeros(N))[1][:,int(N/2)+1]        
            IPRs.append(1/np.sum(np.abs(vectors)**4))
        plt.hist(IPRs)
        plt.xlabel("IPR")
        plt.ylabel("Pogostost")
        plt.title(r"Histogram 1000 izračunanih IPRjev pri $m=0,W=4,N={}$".format(N))
        #plt.savefig(filename)
    
    elif plot=="path":
        seed = 85495455
        #loc length along path
        #========path in phase space=========
        if 1:
            #straight right over phase transition
            ms = np.zeros(200)
            #Wji = np.linspace(3.8+0.65*0.4,3.8+0.75*0.4,200)
            Ws = np.linspace(3.8,4.2,200)
        if 0:
            #straight up tangentially
            ms =np.linspace(-0.1,0.1,1000)
            Ws = np.ones(1000)*4
        if 0:
            #straight up at low disorder
            ms = np.linspace(0.9,1.1,1000)
            Ws = np.ones(1000)
        if 0:
            #tangentially on tilted part
            ms = np.linspace(0.9,1.02,1000)
            Ws = np.linspace(3.81,3.72,1000)            
        
        if 0:
            #visual repr. of path
            mss = np.linspace(-2,2,100)
            Wss = np.linspace(1,5,100)
        
            result= np.zeros((100,100))
            for i in range(100):
                for j in range(100):
                    result[i][j] = lengthForula(mss[i],Wss[j])
                    
            cnt = plt.contourf(Wss,mss,result,levels=np.linspace(np.amin(result),np.amax(result),50))
            
            plt.plot(Wji,mji,color="r")
        if 1:
            #energies on path
            fig,ax = plt.subplots(2)
            Ns = [N]
            for N in Ns:
                print(N)
                np.random.seed(seed)
                omega1 = np.random.rand(N)-0.5
                omega2 = np.random.rand(N)-0.5
                length = []
                energies = []
                vectors = []
                for t in range(200):
                    print(t)
                    vs = np.ones(N)*ms[t] + Ws[t]*omega2
                    ws = np.ones(N)+0.5*Ws[t]*omega1
                    res = constructHPBC(N,vs,ws,np.zeros(N))
                    energies.append(np.abs(res[0][int(N/2):int(N/2)+4].astype(np.complex128)/factor))
                    length.append(lengthEmpiric(vs[1::2],ws[::2]))
                    vectors.append(res[1][:,int(N/2)-1])
                    
            ax[0].plot(np.linspace(0,1,200),length)
            ax[0].set_title(r"$\Lambda$, seed: " + str(seed))
            ax[0].set_yscale("log")
            for i in range(4):
                ax[1].plot(np.linspace(0,1,200),[en[i] for en in energies])
            ax[1].set_title("4 najmanjše pozitivne energije")
            ax[1].set_yscale("log")
            
            
            """
            POSSIBLY ALL IRRELEVANT HERE
            ax[0][0].plot(np.linspace(0,1,200),dolzina)
            ax[0][0].set_title(r"$\Lambda$")
            for i in range(7):
                ax[1][0].plot(np.linspace(0,1,200),[en[i] for en in energije])
            ax[1][0].set_title("7 energij najbližje ničelni.")
            ind = np.argmax(dolzina)
            ax[0][1].plot(np.arange(1,1001),np.abs(vektorji[ind]))
            ax[0][1].set_title(r"Vektor z energijo 0 v maksimumu $\Lambda$")
            ax[1][1].plot(np.arange(1,1001),np.abs(vektorji[ind-10]))
            ax[1][1].set_title(r"Vektor z energijo 0 10 korakov pred maksimumom")
            """
            #plt.plot(np.linspace(0,1,1000),[dolzinaPoFormuli(mji[i],Wji[i]) for i in range(1000)],label=r"Formula")
            #plt.xlabel(r"$t$")
            plt.tight_layout()
            #plt.savefig("vikendtest/tretjic/20.pdf")
            #plt.ylabel(r"$\Lambda$")
            #plt.legend(loc="best")
            
#localization(1000,"IPR","Figures/locLength.pdf")
    
def timeEvolution(N,filename):
    """excitations during time evolution"""
    seed = 85495455
    np.random.seed(seed)
    omega1 = np.random.rand(N)-0.5
    omega2 = np.random.rand(N)-0.5
    
    Winitial = 3.5  #phase transition at W=4
    Wfinal = 4.5   # W(t) = Winitial + (Wfinal-Winitial)/T * t         
    T=100 #quench time
    taus=[1] #timestep
    allpsis=[]
    for tau in taus:
        print(tau)
        steps = int(T/tau)
    
        ms = np.zeros(steps)
        Ws = np.linspace(Winitial,Wfinal,steps)
    
        vs = np.ones(N)*ms[0] + Ws[0]*omega2
        ws = np.ones(N)+0.5*Ws[0]*omega1
        psis = constructHPBC(N,vs,ws,np.zeros(N))[1][:,:int(N/2)]
        for i in range(1,steps):
            vs = np.ones(N)*ms[i] + Ws[i]*omega2
            ws = np.ones(N)+0.5*Ws[i]*omega1
            b = np.matmul(timeEvolutionMatrix(N,0.5*tau,vs,ws).todense(),psis)
            psis = lins.spsolve(timeEvolutionMatrix(N,-0.5*tau,vs,ws).tocsc(),b)
        allpsis.append(psis)
    
    
#    final = constructHPBC(N,vs,ws,np.zeros(N))[1][:,int(N/2):]
#    P0 = np.zeros((N,N),dtype=np.complex128) 
#    for i in range(int(N/2)):
#        P0 = P0 + np.tensordot(np.conjugate(final[:,i]),final[:,i],0)
#    res=[]
#    for i in range(3):
#        P = np.zeros((N,N),dtype=np.complex128)
#        for j in range(int(N/2)):
#            P = P + np.tensordot(np.conjugate(allpsis[i][:,j]),allpsis[i][:,j],0)
#        res.append(np.trace(np.matmul(P,P0)))
#
#    plt.plot(taus,res,"o")
#    plt.ylabel(r"$N_{ex}$")
#    #plt.yscale("log")    
#    plt.xlabel(r"$\tau$")
#    plt.title(r"Število eksitacij v odv. od $\tau$. Seed 85495455, T=100")
#    plt.savefig(filename)
    
    
    #ANIMATION FOR TESTING STUFF OUT. IRRELEVANT NOW
#    if 1: #animacija za test
#        
#        fig, ax = plt.subplots()
#        x = np.arange(1,N+1)
#        ln, = plt.plot([], [])
#        
#        def update(frame):
#            print(frame)
#            ax.clear()
#            ln.set_data(x, np.abs(psis[frame]))
#            W = Winitial + (Wfinal-Winitial)*100*frame*tau/T
#            ax.set_title(r"W={}".format(W,4))
#            return ln,
#
#        ani = FuncAnimation(fig, update, range(100),interval=100)
#        ani.save("quench3.mp4") 
        
    
def checkStates(N,filename):
    #prerverimo val. funkcije blizu prehoda
    x = np.arange(1,N+1)
    W=4
    m=0
    omega1 = np.random.rand(N)-0.5
    omega2 = np.random.rand(N)-0.5
    W2 = W
    W1 = 0.5*W
    vs = np.ones(N)*m + W2*omega2
    ws = np.ones(N)+W1*omega1
    vectors = constructHPBC(N,vs,ws,np.zeros(N))
    energies = vectors[0]
    vectors = vectors[1]
    fig, ax = plt.subplots(nrows=2,ncols=2)
    ax[0][0].plot(x,np.abs(vectors[:,0]))
    ax[0][0].set_title("Prvo stanje, E={}".format(round(energies[0],4)))
    ax[0][1].plot(x,np.abs(vectors[:,int(N/4)-1]))
    ax[0][1].set_title("N/4-to stanje, E={}".format(round(energies[int(N/4)-1],4)))
    ax[1][0].plot(x,np.abs(vectors[:,int(N/2)-1]))
    ax[1][0].set_title("N/2-to stanje, E={}".format(round(energies[int(N/2)-1],4)))
    print(1/np.sum(np.abs(vectors[:,int(N/2)-1])**4))
    ax[1][1].plot(x,np.abs(vectors[:,int(3*N/4)-1]))
    ax[1][1].set_title("3N/4-to stanje, E={}".format(round(energies[int(3*N/4)-1],4)))
    plt.tight_layout()
    plt.savefig(filename)
    
    
def localizationLength(N,filename):
    ms = np.linspace(-2,2,100)
    Ws = np.linspace(1,5,100)

    result= np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            result[i][j] = lengthFormula(ms[i],Ws[j])
            
    cnt = plt.contourf(Ws,ms,result,levels=np.linspace(np.amin(result),np.amax(result),50))
    plt.colorbar()
    for c in cnt.collections:
        c.set_edgecolor("face")    
     
    plt.xlabel(r"$W$")
    plt.ylabel(r"$m$")
    plt.title(r"$\Lambda^{-1}$")
    plt.savefig(filename)
    
def contourWinding(N,n,filename):
    delta = 4*np.pi / N
    cs = np.array(cTab(n),dtype=np.complex128)
    SPlus = np.diag(np.array([1 if i%2==0 else 0 for i in range(N)],dtype=np.complex128))
    SMinus = np.diag(np.array([0 if i%2==0 else 1 for i in range(N)],dtype=np.complex128))
    X = np.diag(np.repeat(np.arange(int(N/2)),2))
    firstexp = np.array([np.exp(-1j*c*X*delta) for c in range(int(-n/2),int(n/2)+1)],dtype=np.complex128)
    secondexp = np.conjugate(prviexp)
    ms = np.linspace(-2,2,40)
    Ws = np.linspace(0,5,40)
    nus = []
    for m in ms:
        print(m)
        temp = []
        for W in Wji:
            omega1 = np.random.rand(N)-0.5
            omega2 = np.random.rand(N)-0.5
            W2 = W
            W1 = 0.5*W
            vs = np.ones(N)*m + W2*omega2
            ws = np.ones(N)+W1*omega1
            vectors = constructHPBC(N,vs,ws,np.zeros(N))[1]
            temp.append(windingNumber(N,n,vectors,cs,SMinus,SPlus,firstexp,secondexp))
        nus.append(temp)
    nus = np.array(nus)
    cnt = plt.contourf(Ws,ms,nus,levels=np.linspace(np.amin(nus),np.amax(nus),50))
    plt.colorbar()
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.xlabel(r"$W$")
    plt.ylabel(r"$m$")
    plt.title(r"Ovojno število, $n={}, N={}$".format(n,N))
    plt.savefig(filename)


#localization(1000,"IPR","HistogramIPR.pdf")
#localization(1000,"path","HistogramIPR.pdf")
#timeEvolution(1000,"razlika4.pdf")
#checkStates(2000,"StanjaBlizuPrehoda3.pdf")
#localizationLength(1000,"LokalizacijskaDolzina.pdf")
#contourWinding(1000,50,"WindingNumber2.pdf")




#more tests. Irrelevant
if 0:
    #preverimo diference
    Nji = [2*k+1 for k in range(20)]
    matrixRez=[]
    taylorRez=[]
    h=1/10000
    x = np.linspace(-1,1,10000)
    k = np.random.rand()*100
    y = np.sin(k*x)
    tocka = 3342
    pravilno = k*np.cos(k*tocka)
    for N in Nji:
        print(N)
        koef1 = cTab(2*N,h)
        keys = [i for i in range(-N,N+1)]
        koef1Dict = {keys[i] : koef1[i] for i in range(2*N+1)}
        rez=0
        for k in koef1Dict:
            rez+= y[tocka+k]*koef1Dict[k]
        matrixRez.append(rez)
        rez=0
        koef2Dict = dobiCn(N)
        for k in koef2Dict:
            rez+= y[tocka+k]*koef2Dict[k]/h
        taylorRez.append(rez)
    plt.plot(Nji,[abs(i-pravilno) for i in matrixRez],label="Matrika")
    plt.plot(Nji,[abs(i-pravilno) for i in taylorRez],label="Taylor")
    plt.legend()


#Irrelevant below

def arsinhTaylor(N):
    """N red razvoj od arsinh(x)"""
    x=1
    cleni = [x]
    stevilo = int((N-1)/2) #tolko členov moramo vzet
    for k in range(stevilo):
        faktor = -x**2/4* (2*k+1)**2 * (2*k+2)/ ((k+1)**2*(2*k+3))
        cleni.append(faktor*cleni[-1])
    return cleni

def dobiCn(red):
    """red naj bo lih NE POZABI DELIT S H NA KONCU"""
    koeficienti = {i:0 for i in range(-red,red+1)}
    cleni = arsinhTaylor(red)
    for i in range(len(cleni)):
        potenca = int(2*i+1)
        for k in range(potenca+1):
            koeficienti[int(potenca-k-k)]=koeficienti[int(potenca-k-k)]+ (-1)**k * (0.5)**potenca * cleni[i]*binom(potenca,k)
    return koeficienti

def skonstruirajH(N,v,w):
    """Skonstruira in najde lastne rešitve za SSH Hamiltonjan, z hoppingi v in w"""
    dim = 2*N
    diagonalci = np.zeros(dim)
    obdiagonalci = np.array([v if i%2==0 else w for i in range(dim-1)])
    return lin.eigh_tridiagonal(diagonalci,obdiagonalci,check_finite=False)



@jit(nopython=True)
def windingNumberJIT(N,n,vektorji,cji,SMinus,SPlus,prviexp,drugiexp):
    PMinus = np.zeros((N,N),dtype=np.complex128)
    for i in range(int(N/2)):
        PMinus += np.outer(np.conjugate(vektorji[:,i]),vektorji[:,i])
    Q = np.identity(N,np.complex128)-2*PMinus
    QMinusPlus = np.dot(SMinus,np.dot(Q,SPlus))
    QPlusMinus = np.dot(SPlus,np.dot(Q,SMinus))    
    komutator = np.zeros((N,N),dtype=np.complex128)
    for c in range(n+1):
        komutator += cji[c]*np.dot(prviexp[c],np.dot(QPlusMinus,drugiexp[c]))
    return 1/(0.5*N)*np.trace(np.dot(QMinusPlus,1j*komutator))
