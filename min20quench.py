from matplotlib import animation
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
import mpmath as mp 
import scipy.sparse as sparse

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


def makeTheFile():
    """20 lowest positive energy eigenfunctions"""
    seed = 10158825
    np.random.seed(seed)
    N=1000
    omega1 = np.random.rand(N)-0.5
    omega2 = np.random.rand(N)-0.5
    Winitial = 3.5  #fazni prehod okoli 3.9
    Wfinal = 4.5   # W(t) = Winitial + (Wfinal-Winitial)/T * t        
    Ws = np.linspace(Winitial,Wfinal,100)
    res = []
    for W in Ws:
        vs = W*omega2
        ws = np.ones(N)+0.5*W*omega1
        energies = constructHPBC(N,vs,ws,np.zeros(N))[1][:,int(N/2):int(N/2)+20]
        res.append(energies)
        

    np.save("min20.npy",np.array(res))
    
    
def useTheFile1():
    """IPR(t)"""
    res = np.load("min20.npy")
    times = np.linspace(3.5,4.5,100)
    IPRs=np.zeros((5,100))
    for i in range(100):
        for j in range(5):
            IPRs[j][i] = 1/np.sum(np.abs(res[i][:,j])**4)
    
    for j in range(5):
        plt.plot(times,IPRs[j],label=str(j))
    plt.legend()
    
useTheFile1()
    

def useTheFile2():
    """Make animation of the wavefunctions during quench"""
    res = np.load("min20.npy")
    fig, ax = plt.subplots()    
    x = np.arange(1,1001)
    def animate(frame):
        ax.clear()
        for i in range(2):
            ax.plot(x,np.abs(res[frame][:,i])**2)
        ax.set_title("W={}".format(round(3.5+frame/100,2)))
        ax.set_ylim(ymin=0,ymax=1)
    ani = animation.FuncAnimation(fig,animate,list(range(100)))
    ani.save("min20.mp4")
