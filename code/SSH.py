
import numpy as np
import scipy.linalg as lin
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri

sirinca=434.62277*0.0138
visinca=320.11156*0.0138
plt.rc('text', usetex=True)
plt.rc('font', size=12)
plt.rc('figure', figsize=(sirinca, visinca))

def constructH(N,v,w):
    """Constructs and finds eigenfunctions of the SSH hamiltonian with N atoms and hoppings v,w"""
    diagonals = np.zeros(N)
    updiagonals = np.zeros(N-1)*1j
    updiagonals[::2] = v[1::2]*1j
    updiagonals[1::2] = w[2::2]
    updiagonals = np.insert(updiagonals,0,0)
    A = np.vstack((updiagonals,diagonals))
    return lin.eig_banded(A,overwrite_a_band=True,eigvals_only=True)

def constructHPBC(N,v,w,epsilons):
    """periodic boundary conditions"""
    if np.isscalar(v): #Actually don't need N v-s. One remains unused.
        v = np.ones(N)*v
        w = np.ones(N)*w
    
    #N epsilons on diagonal,      odd v*(-1j) on odd places, even w on even spaces, first w goes in corner
    def H(i,j):
        if i==j:
            return 0
            return epsilons[i]
        if j==i+1:
            return 0
            if i%2==0:
                return v[i]*1j
                return v[i]
            else:
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
            #return w[-1] TLE JE PONAVADI TO, AMPAK ZA QUENCHANO SEM MOGU W[0] OČITNO
        else:
            return 0
        
    matrix = np.array([[H(j,i) for i in range(N)] for j in range(N)])
    return lin.eigh(matrix) #eigh only uses lower triangle

def loc(vs,ws):
    tns = np.concatenate((ws[2::2],np.array([ws[-1]])))
    ms = vs[1::2]
    lamb = np.sum(np.log(np.abs(tns))-np.log(np.abs(ms)))/len(ms)
    return 1/np.abs(lamb)


def IPRContour(v,w,N,filename):
    """controuf plot of IPR with respect to randomness for many energies"""
    #deltas = [0.1*i for i in range(51)] #SSH
    deltas = [0.01*i for i in range(110)] #anderson
    energies = []
    IPRs = []
    epsilons = []
    np.random.seed(100000)
    for delta in deltas:
        print(delta)
        #SSH
        #W2 = delta
        #W1 = 0.5*delta
        #ws = 1+W1*(np.random.rand(N)-0.5)
        #vs = W2*(np.random.rand(N)-0.5)
        
        #ANDERSON
        #epsilon = 0.5*delta*(2*np.random.rand(N)-1)
        vs = 1+delta*(2*np.random.rand(N)-1)
        ws = 1+delta*(2*np.random.rand(N)-1)
        
        #vs = 0.5+delta*(2*np.random.rand(N)-1)
        #ws = 1+0.5*delta*(2*np.random.rand(N)-1)
        result = constructHPBC(N,vs,ws,np.zeros(N))
        if delta==0.03 and 1==0:
            aa = N-1
            print(result[0][aa])
            print(1/np.sum(np.abs(result[1][:,aa])**4))
            plt.plot(np.arange(1000),np.abs(result[1][:,aa])**2)
            č
        energy, indices = np.unique(result[0],return_index=True)
        IPR = [1/np.sum(np.abs(result[1][:,i])**4) for i in range(N) if i in indices]
        IPRs = IPRs + IPR #conatenation
        energies   = energies + energy.tolist()
        epsilons = epsilons + [delta for i in range(len(IPR))]

#   SSH
    energies = np.array(energies)
    epsilons = np.array(epsilons)
    trian = tri.Triangulation(energies,epsilons)
    triangles = trian.triangles
  
#    mask = np.zeros(len(triangles))
#    for i in range(len(triangles)):
#        tr = triangles[i]
#        for j in range(3):
#            if abs(abs(energies[tr[j]])-2)<1e7 and epsilons[tr[j]]<1e3:
#                if (epsilons[tr[(j+1)%3]])
#                mask[i]=1
#                break
#            if epsilons[tr[j]] < 1.79:
#                if abs(energies[tr[j]])<0.166:
#                    mask[i]=1
#                    break
#            x0 = energies[tr[j]]
#            x1 = energies[tr[(j+1)%3]]
#            if x0 < 0 and x1 > 0:
#                y0 = epsilons[tr[j]]
#                y1 = epsilons[tr[(j+1)%3]]
#                t= x1/(x1-x0)
#                if y0*t+(1-t)*y1 < 2.7: #prej 2.35
#                    mask[i]=1
#                    break
                
    xtri = energies[triangles] - np.roll(energies[triangles], 1, axis=1)
    ytri = epsilons[triangles] - np.roll(epsilons[triangles], 1, axis=1)
##    trian.set_mask(np.abs(np.abs(xtri)-2)<1e8 and ytri<0.01)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    trian.set_mask(maxi > 0.2)
    #trian.set_mask(mask)
    #plt.triplot(trian)
    #plt.scatter(energies,epsilons,c=[i/1000 for i in IPRs],s=15,cmap="viridis")
    cnt = plt.tricontourf(trian,IPRs,levels=np.linspace(0,1000,50))
    #cnt = plt.tricontourf(energies,epsilons,IPRs,levels=np.linspace(0,1000,50),antialiased=False)
#
    for c in cnt.collections: #remove ugly  white lines
        c.set_edgecolor("face")
    cb = plt.colorbar(cnt)
    cb.set_ticks([i*100 for i in range(11)])
    cb.set_ticklabels([i*100 for i in range(11)])
    plt.ylim(0,1)
    plt.xlabel(r"$E$")
    plt.ylabel(r"$W$")
    #plt.title(r"SSH z neredom v sklopitvah, IPR".format(N))
    plt.savefig(filename)
#IPRContour(0.5,1,1000,"SSHIPRNered.pdf")
#IPRContour(0,0,1000,"SSHIPR2.pdf")
def plotWaveFunctions(N):
    """For masters. Wavefunction plots as we get near critical point"""
    x=np.arange(1,N+1)
    fig, axi = plt.subplots(1,2,sharey=True)
    #deltas = [3.8+0.001*i for i in range(201)]
    #deltas = np.linspace(3.8,4.2,10000)
    np.random.seed(8166247)
    rand1 = np.random.rand(N)
    rand2 = np.random.rand(N)
    delta1=3.8
    delta2=3.97
    for delta in [delta1,delta2]:
        W2 = delta
        W1 = 0.5*delta
        ws = 1+W1*(rand1-0.5)
        vs = W2*(rand2-0.5)
        
        #epsilon = 0.5*delta*(2*np.random.rand(N)-1)
        #vs = 1+delta*(2*np.random.rand(N)-1)
        #ws = 1+delta*(2*np.random.rand(N)-1)
        
        #vs = 0.5+delta*(2*np.random.rand(N)-1)
        #ws = 1+0.5*delta*(2*np.random.rand(N)-1)
        
        #lok = loc(vs,ws)
        
        if delta==delta1:
            result = constructHPBC(N,vs,ws,np.zeros(N))
            #print(result[0][int(N/2)])
            axi[0].plot(x,np.abs(result[1][:,int(N/2)])**2,color="blue",label="Prvo stanje")
            axi[0].plot(x,np.abs(result[1][:,int(N/2)+1])**2,color="orange",label="Drugo stanje")
            axi[0].plot(x,np.abs(result[1][:,int(N/2)+2])**2,color="red",label="Tretje stanje")
            axi[0].set_ylabel(r"$|\Psi|^2$")
            axi[0].set_xlabel(r"$x$")
            axi[0].legend()
            axi[0].set_title(r"$W={}$".format(delta1))
            axi[0].grid()
        if delta==delta2:
            result = constructHPBC(N,vs,ws,np.zeros(N))
            #print(result[0][int(N/2)])
            axi[1].plot(x,np.abs(result[1][:,int(N/2)])**2,color="blue")
            axi[1].plot(x,np.abs(result[1][:,int(N/2)+1])**2,color="orange")
            axi[1].plot(x,np.abs(result[1][:,int(N/2)+2])**2,color="red")
            axi[1].set_title(r"$W=4$")
            axi[1].set_xlabel(r"$x$")
            axi[1].grid()
    plt.tight_layout()
    plt.savefig("EigenPloti.pdf")
#plotWaveFunctions(1000)

def DOSHist(N,filename):
    W=2.8
    vs = 1+0.5*W*(np.random.rand(N)-0.5)
    ws = W*(np.random.rand(N)-0.5)
    energies = [i for i in constructH(N,vs,ws)[int(N/2):] if i<0.01]
    vred,bini,lol = plt.hist(energies,bins=100,color="C0")
    
    def model(x,A):
        return A/(np.abs(x)*np.log(np.abs(x))**3)
    x = bini[:-1]+0.5*(bini[1]-bini[0])
    A = vred[1]*np.abs(x[1])*np.log(np.abs(x[1]))**3
    x=np.linspace(x[0]-4e-5,x[-1],1000)
    y = [model(i,A) for i in x]
    plt.plot(x,y,label=r"$N \propto \frac{1}{E\  \log^3 E}$",color="k")
    
    plt.hist([-i for i in energies],-bini[::-1],color="C0")
    plt.plot(-x,y,color="k")
    plt.xlabel(r"$E$")
    plt.xticks([-0.01,-0.005,0,0.005,0.01])
    plt.ylabel(r"$N$")
    plt.legend(loc="center left")

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
#DOSHist(150000,0)
    
    

def DOS(v,w,N,filename,toplot="subplots"):
    #density of states

    def cauchy(x,x0):
        """cauchy/lorentz distribution to approximate dirac delta"""
        HWHM = 0.01
        return 1/np.pi * HWHM / ((x-x0)**2 +HWHM**2) 
    
    def density(x,states):
        suma=0
        for r in states:
            suma += cauchy(x,r)
        return suma

    if toplot=="subplots":
        #DOS for some values of delta
        #delte = [0,0.5,1,1.5,2,2.5,3,3.5,4]
        deltas = [0.1*i for i in range(9)]
        fig, axi = plt.subplots(3,3,sharey=True)
        axi = axi.flatten()
        for i in range(9):
            print(i)
            delta = deltas[i]
            vs = 1+0.5*delta*(2*np.random.rand(N)-1)
            ws = 1+0.5*delta*(2*np.random.rand(N)-1)
            result = constructHPBC(N,vs,ws,np.zeros(N))[0]
            es = np.linspace(np.amin(result),np.amax(result),100)
            difference = es[1]-es[0]
            res = np.array([density(e,rez) for e in es])
            axi[i].plot(es,2*N*res/(np.sum(res)*difference)) #hmm
            axi[i].set_title("$\Delta = {}$".format(round(deltas[i],2)))
        plt.tight_layout()
        plt.savefig(filename)
    
        
    elif toplot=="one":
        #several DOS on one plot
        deltas = [0.05*i for i in range(15)]
        cmap = plt.get_cmap("brg")
        colors = np.linspace(0,1,15)
        lower = 0
        fig,ax=plt.subplots()
        for i in range(15):
            print(i)
            delta = deltas[i]
            vs = 1+0.5*delta*(2*np.random.rand(N)-1)
            ws = 1+0.5*delta*(2*np.random.rand(N)-1)
            res = constructHPBC(N,vs,ws,np.zeros(N))[0]
            es = np.linspace(np.amin(res),np.amax(res),100)
            difference = es[1]-es[0]
            result = np.array([density(e,res) for e in es])
            result = 2*N*result/(np.sum(result)*difference)
            lower+= np.amax(result)+1
            ax.plot(es,result+np.ones(100)*lower,color=cmap(colors[i]),label=r"$\Delta={}$".format(round(delta,2)))        
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(r"$E$")
        ax.legend(bbox_to_anchor=(1.05,0.5),loc="center")
        ax.set_title(r"Gostota stanj za SSH model z disorderjem v sklopitvi, $N={}$".format(N))
        plt.savefig("GostoteStanjSSHDisorderV2.pdf")

def dispersion(v,N,filename):
    """dispersion changing w"""
    ws =np.linspace(0.8,1,20)
    cmap = plt.get_cmap("plasma")
    colors = np.linspace(0,1,20)
    fig,ax = plt.subplots()
    for i in range(20):
        w = ws[i]
        ks = np.linspace(-np.pi,np.pi,int(N/2))
        res = constructHPBC(N,v,w,np.zeros(N))[0]
        band1 = res[:int(N/2)][::-1]
        band2 = res[int(N/2):] 
        energies1 = np.concatenate((band1[1:-1][::2][::-1],[band1[0]],band1[1:-1][1::2],[band1[-1]]))
        energies2 = np.concatenate((band2[1:-1][::2][::-1],[band2[0]],band2[1:-1][1::2],[band2[-1]]))
        if i in [0,4,9,14,19]:
            ax.plot(ks,energies1,color=cmap(colors[i]),label=r"$w={}$".format(round(w,3)))
        else:
            ax.plot(ks,energies1,color=cmap(colors[i]))
        ax.plot(ks,energies2,color=cmap(colors[i]))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E$")
    ax.legend()
    #ax.legend(bbox_to_anchor=(1,0.5),loc="center")
    ax.set_title(r"Disperzija SSH, $N={}, v={}$".format(N,v))
    plt.savefig(filename)


#expontentialDecayCheck(0.5,1,"SSHEn.pdf")
def exponentialDecayCheck(v,w,filename):
    """Check the formula claiming energy falls exponentially with N"""
    Ns = np.array([5+i for i in range(16)])
    res= []
    for N in Ns:
        temp = constructH(N,v,w)
        print(temp[0][N])
        res.append(temp[0][N])
    plt.plot(Ns,res,color="g",label=r"Diagonalizacija")
    ksi = 1/np.log(w/v)
    
    def model(x,A):
        return A*np.exp(-(x-1)/ksi)
    A, = curve_fit(model,Ns,res)[0]    
    plt.plot(Ns,A*np.exp(-(Ns-1)/ksi),ls="--",color="r",label=r"$E =A e^{-(N-1)/\xi}$")
    plt.legend()
    plt.title(r"$w=1, v=0.5$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$E$")
    plt.savefig(filename)
    
def energies(N,filename):
    """ Energy plots changing v"""
    vs = np.linspace(0,2,50)
    result = np.ones((50,2*N))*1.0
    for i in range(50):
        result[i] = constructH(N,vs[i],1)[0]
        
    cmap = plt.get_cmap("brg")
    #colors = np.linspace(0,1,21)
    for j in range(4):
        plt.plot(vs,result[:,N-2+j],color="k")
    plt.xlabel(r"$v$")
    plt.ylabel(r"$E$")
    #plt.title(r"$E(v), w=1, N={}$".format(N))
    #plt.savefig(filename)
    
#energies(1000,"lol")

def eigenfunctionPlots(filename):
    """ Check the plots of the 3 wavefunctions in the article """
    res = constructH(333,0.5,1)[1]
    first = res[:,7]
    second = res[:,333]
    third = res[:,332]
    xs = [0.5+i for i in range(0,666)]
    fig,ax = plt.subplots(3,sharex=True)
    ax[0].bar(xs,first,width=0.5,tick_label=["A" if i%2==0 else "B" for i in range(20)])
    ax[1].bar(xs,second,width=0.5,tick_label=["A" if i%2==0 else "B" for i in range(20)])
    ax[2].bar(xs,third,width=0.5,tick_label=["A" if i%2==0 else "B" for i in range(20)])
    plt.suptitle("Lastni vektorji za 8.,10. in 11. lastno stanje")
    plt.savefig(filename)
    

def animation(N,v,filename):
    """animation for eigenfunctions close to zero as we change w"""
    fig,[ax1,ax2] = plt.subplots(2,figsize=(10,15))
    N=333
    v = 1
    x = [i for i in range(1,2*N+1)]
    ws= [0.05*i for i in range(31)]
    def animiraj(frame):
        print(frame)
        ax1.clear()
        ax2.clear()
        w = ws[frame]
        a = constructH(N,v,w)[1]
        vector= a[:,N-1]
        ax1.plot(x,vector/scipy.linalg.norm(vector))
        ax1.set_title("N-to lastno stanje")
        vektor = a[:,N]
        ax2.plot(x,vector/scipy.linalg.norm(vector))
        ax2.set_title("N+1. lastno stanje")
        plt.suptitle(r"$w/v = {}$".format(round(w/v,2)))
    ani = FuncAnimation(fig,animiraj,range(26),interval=333)
    ani.save(filename)
    
    
    
    
    
    
    
    
    
    
    

#IPRContour(1,0.5,2000,"IPRSSHDisorderV.pdf")
#    
#DOS(1,0.5,1000,"GostoteStanjSSHDisorderV.pdf","subplots")
#DOS(1,0.5,1000,"GostoteStanjSSHDisorderV2.pdf","one")
#dispersion(1,1000,"SSHDisperzija.pdf")
#animation(333,1,"SSH.mp4")
#eigenfunctionPlots("SSHFunkcije.pdf")
#energies(10,"SSHEnergije.pdf")
