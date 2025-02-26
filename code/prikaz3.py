import matplotlib.pyplot as plt
import os
import numpy as np
import re
from matplotlib import animation
from scipy.optimize import curve_fit
from numpy.fft import fft, ifft

def model(x,A,B,C):
    return A*x**B*np.log(x)**C

def periodic_corr(x,y):
    rez = ifft(fft(x)*fft(y).conj()).real
    return rez/rez[0]

#sirinca=434.62277*0.0138
#visinca=426.797*0.0138
plt.rc('text', usetex=True)
plt.rc('font', size=10)
#plt.rc('figure', figsize=(sirinca, visinca))

#files = [i for i in os.listdir("./EksitacijeVecjiSampleRate")]
#files = [i for i in os.listdir("./EksitacijeSproti")]
N = 1000
pattern1 = re.compile(r"\[[^[]*\]")
#pattern = re.compile(r"\(\d+\.\d+[+-]")

def getEverything(foldername):
    everything=[]
    files = [i for i in os.listdir("../{}".format(foldername))]
    for file in files: 
        f = open("../{}/".format(foldername) + file,"r")

        data = re.findall(pattern1,f.read())
        every = np.zeros((len(data),500))*1j
        for j in range(len(data)):
            t = list(map(complex,data[j][1:-1].split(","))) #excitations during/after
            #t = list(map(complex,data[j][1:-1].split())) #spageti
            #print(len(t))
#            except:
#                print(file)
#                f.close()
#                [float(w) for w in data[j][1:-1].split(",")]
            every[j] += np.array(t)
            #every[j]+=[t[i]+t[i+1] for i in range(0,1000,2)]
        everything.append(every)
        f.close()
    return np.array(everything)
    
def FinalExcPlotsMasters(ax,hitrost=False):
    
    quench3545 = [10,30,100,300,1000,3000,10000]
    quench35 = [20,60,200,600,2000,6000,20000]
    quench26 = [40,90,400,900,4000,9000,40000]

#    quench3545 = [300,1000,3000,10000]
#    quench35 = [600,2000,6000,20000]
#    quench26 = [900,4000,9000,40000]
    
    v = {1:1,2:2,3:4}
    eks = {1:(quench3545,[],r"$W \in [3.5,4.5]$"),2:(quench35,[],r"$W \in [3,5]$"),3:(quench26,[],r"$W \in [2,6]$")}
    barve = ["blue","orange","green"]
    for key in eks:
        for q in eks[key][0]:
            everything = getEverything("../ExcByEnergy/eksitacije/{}".format(q))
            everything = np.sum(everything,axis=0)/len(everything)
            eks[key][1].append(sum(everything[-1]))
            
        x = eks[key][0]
        y = eks[key][1]
        fit = curve_fit(model,x,y)
        A,B,C = fit[0]
        
        if hitrost:
            x = v[key]/np.array(eks[key][0])
            
        ax.plot(x,y,lw=0,marker=".",label=eks[key][2])            
        x=np.linspace(x[0],x[-1],1000)

        if hitrost:
            ax.plot(x,[model(i**(-1)*v[key],A,B,C) for i in x],ls=":",color=barve[key-1])
        else:
            ax.plot(x,[model(i,A,B,C) for i in x], ls=":", color=barve[key-1])
            y = model(x[0],A,B,C)
            if key==1:
                ax.text(x[0],y-6,r"$N_{eks} \propto T^{%.3f}  (\log T)^{%.3f}$" % (B,C),fontsize=15)
            elif key==2:
                ax.text(x[0],y-2,r"$N_{eks} \propto T^{%.3f}  (\log T)^{%.3f}$" % (B,C),fontsize=15)                
            elif key==3:
                ax.text(x[0],y,r"$N_{eks} \propto T^{%.3f}  (\log T)^{%.3f}$" % (B,C),fontsize=15)
        
    if hitrost:
        ax.set_xlabel(r"$v$",fontsize=15)
        #ax.set_title("Število eksitacij po quenchu v odvisnosti od hitrosti le-tega.",fontsize=12)
    else:
        ax.legend(fontsize=14)
        ax.set_xlabel(r"$T$",fontsize=15)
        #ax.set_title("Število eksitacij po quenchu v odvisnosti od dolžine le-tega.",fontsize=12)
        ax.set_ylabel(r"$N_{eks}$",fontsize=15)
        
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()

def FinalExcThruTimeMasters(ax,which):
    barve = [plt.get_cmap("gist_rainbow")(i) for i in np.linspace(0,1,7)]
    if which==1:
        x = np.linspace(3.5,4.5,10)
        quench = [10,30,100,300,1000,3000,10000]
        f1=0.168
        f2=0.029
    elif which==2:
        x = np.linspace(3,5,20)
        quench = [20,60,200,600,2000,6000,20000]
        f1=0.141
        f2=0.454
    elif which==4:
        x = np.linspace(2,6,40)
        quench = [40,90,400,900,4000,9000,40000]
        f1=0.096
        f2=0.878
    hitrosti = [0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]
    for q,barva,hitrost in zip(quench,barve,hitrosti):
        everything = getEverything("../ExcByEnergy/eksitacije/{}".format(q))
        everything = np.sum(everything,axis=0)/len(everything)
        everything = np.sum(everything,axis=1)
        #ax.plot(x,everything,label=r"$v={}$".format(hitrost), color=barva)
        ax.plot((x-4)*q**f1*np.log(q)**f2 + 4,everything/everything[-1],label=r"$v={}$".format(hitrost),color=barva)
        
    ax.grid()
    ax.set_xlabel(r"$\tilde{W}$",fontsize=15)
    ax.set_ylabel(r"$\tilde{N}_{eks}$",fontsize=15)
    ax.set_title(r"$W \in [{},{}]$".format(str(x[0]),str(x[-1])))


#fig,axi = plt.subplots(1,3,figsize=(12,6))
#FinalExcThruTimeMasters(axi[0],1)
#FinalExcThruTimeMasters(axi[1],2)
#FinalExcThruTimeMasters(axi[2],4)
#axi[1].legend(fontsize=14)
#plt.tight_layout()
#plt.savefig("SkoziCasAlt.pdf")


fig, axi = plt.subplots(1,2,sharey=True,figsize=(12,6))
FinalExcPlotsMasters(axi[0])
FinalExcPlotsMasters(axi[1],True)
plt.tight_layout()
plt.savefig("Skaliranje3Alt.pdf")

def forT2000():
    files = [i for i in os.listdir("./vecEksFiksenT/T2000")]
    res = np.zeros(2000)
    for file in files:
        f = open("./vecEksFiksenT/T2000/" + file).read()
        data = np.real(np.array(list(map(complex,f.split()))))
        res += data
    #return res[-1]/len(files)
    plt.plot(np.linspace(3,5,2000),res/len(files),label="T=2000")
    #plt.plot((np.linspace(3,5,2000)-4)/10**(-0.25)+4,res/len(files)/2000**(-0.25),label="T=2000")


 
def forT40Extended():
    files = [i for i in os.listdir("./vecEksFiksenT/T40")]
    res = np.zeros(40)
    for file in files:
        f = open("./vecEksFiksenT/T40/" + file).read()
        data = np.real(np.array(list(map(complex,f.split()))))
        res += data
    plt.plot(np.linspace(2,6,40),res/len(files),label="T=40")
    
#x = np.arange(500)

#forT40Extended()

#10 340 670 1000
#10 46 215 1000

def ExcitationForMoreT(x,every):
    fig, axi = plt.subplots(2,2,sharex=True,sharey=True)
    axi[0][0].bar(x,every[0])
    axi[0][0].set_title("T=10")
    axi[0][0].set_ylabel(r"$N_{ex}$")
    axi[0][1].bar(x,every[20])
    axi[0][1].set_title("T=46")
    axi[1][0].bar(x,every[40])
    axi[1][0].set_title("T=215")
    axi[1][0].set_ylabel(r"$N_{ex}$")
    axi[1][0].set_xlabel(r"$x$")
    axi[1][1].bar(x,every[80])
    axi[1][1].set_title("T=1000")
    axi[1][1].set_xlabel(r"$x$")
    plt.savefig("HistogramiLog.pdf")

def ExcitationsDuring():
    every = getEverything("../zaHist")[0] #T=6000 3->5
    x = np.arange(1,501)
    fig, axi = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,10))
    axi[0][0].bar(x,every[0])
    axi[0][0].set_title(r"$W=3.5$")
    axi[0][0].set_ylabel(r"$N_{ex}$")
    axi[0][1].bar(x,every[1])
    axi[0][1].set_title(r"$W=4$")
    axi[1][0].bar(x,every[2])
    axi[1][0].set_title(r"$W=4.5$")
    axi[1][0].set_ylabel(r"$N_{ex}$")
    axi[1][0].set_xlabel(r"$x$")
    axi[1][1].bar(x,every[3])
    axi[1][1].set_title(r"$W=5$")
    axi[1][1].set_xlabel(r"$x$")
    for i in range(4):
        print(sum(every[i]))
    #axi[0][0].set_yscale("log")
    #axi[0][1].set_yscale("log")
    #axi[1][0].set_yscale("log")
    #axi[1][1].set_yscale("log")
    #plt.suptitle(r"Krajevna porazdelitev eksitacij. $v=0.01, W: 3.5 \rightarrow 4.5$")
    plt.savefig("KrajevneEksitacijeNoGrid.pdf")

#ExcitationsDuring()

def ExcitationsAnimation(every):  
    fig, ax = plt.subplots()    
    
    def animate(frame):
        toplot = every[frame]
        ax.clear()
        ax.bar(x,toplot)
        ax.set_ylim(ymin=0,ymax=0.5)
        ax.set_title("W={}".format(round(3.5+frame/100,2)))
    ani = animation.FuncAnimation(fig,animate,list(range(100)))
    ani.save("EksitacijeSkoziQuenc.mp4")

def numOfExc(everything,T=""):
    n= len(everything[0])
    everyy = np.zeros(n)
    for every in everything:
        everyy += np.real(np.array([sum(i) for i in every]))
    #return everyy[-1]/99
    plt.plot(np.linspace(3,5,n),everyy/99,label="T={}".format(T))
    #plt.plot(np.linspace(3,5,n),everyy/99*T**(0.25),label="T={}".format(T))

def autocorrelation(every,x):
    """rescaled autocorrelation"""
    for i in range(4):
        every[i] -= np.mean(every[i])
    k1 = 0
    k = 0
    x = x[:250]
    autocor = periodic_corr(every[0],every[0])[:250]
    plt.plot(x*10**k1,autocor*10**k)
    
    autocor = periodic_corr(every[1],every[1])[:250]
    plt.plot(x*46**k1,autocor*46**k)

    autocor = periodic_corr(every[2],every[2])[:250]
    plt.plot(x*215**k1,autocor*215**k)

    autocor = periodic_corr(every[3],every[3])[:250]
    plt.plot(x*1000**k1,autocor*1000**k)    


def IPROfT(everything):
    
    def model(x,A,B,C):
        return A*x**B+C
    
    times = np.linspace(10,1000,50)
    AvgIPR = np.zeros(50)
    for every in everything:
        IPRs = []
        for i in range(50):
            probs = every[i]/sum(every[i])
            IPRs.append(1/sum([v*v for v in probs]))
        AvgIPR+=np.real(np.array(IPRs))
    plt.plot(1/times,AvgIPR/99,label="Rezultat")
    A,B,C = curve_fit(model,1/times,AvgIPR/99)[0]
    print(A,B,C)
    plt.plot(1/times,A*(1/times)**B+C,"--",color="k",label="Fit")
    plt.text(0.02,120,r"$N_{ex} = 224*v^{-0.17} + 20.5$")
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$v$")
    plt.ylabel("IPR")
    plt.legend()
    plt.title(r"IPR končnega profila eksitacij, $W \in [3.5,4.5]$")

    plt.savefig("Figures/EksIPR.pdf")
#IPROfT(getEverything("EksitacijeVecjiSampleRate"))
    
    
#eks=[]
#
#for T in [20,200,2000]:
#    print(T)
#    if T==2000:
#        forT2000()
#        continue
#    everything = getEverything("vecEksFiksenT/T{}".format(str(T)))
#    numOfExc(everything,T)
#č   
#plt.plot([20,200,2000],eks)
#plt.yscale("log")
#plt.xscale("log")
#
#plt.legend()
#plt.xlabel(r"$W$")
#plt.ylabel(r"$N_{ex}$")
#plt.title("Število eks. od W za več časov quencha")
#    
#    
#    
#    
    
   
    
    
    
    
    
    
    
    
    
    
    
#def model(x,A,B):
#    return A + B/(abs(x)*np.log(abs(x)))
#
#def lok(x):
#    W = x+4
#    num = (2+0.5*W)**(2/W+0.5)*W**(-0.5)
#    den = abs(2-0.5*W)**(2/W-0.5)*W**(0.5)
#    return 1/np.abs(np.log(num/den))
#
#x = -np.linspace(0.01,0.1,100)
#y = np.vectorize(lok)(x)
#
#A,B = curve_fit(model,x,y)[0]
#yy = [model(i,A,B) for i in x]
#
#plt.plot(x,y,"k")
#plt.plot(x,yy,"--",color="b")
#    
#    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
""" CONTOUR PRIKAZ
t = vse[0]
#t = [sum(t[i:i+10]) for i in range(0,500,10)]
zaplottat = np.vstack((t,t,t,t,t))
plt.contourf(zaplottat,levels=np.linspace(np.amin(t),np.amax(t),50))
plt.colorbar()
"""    

"""
def lok(W):
    W2 = W
    W1 = 0.5*W
    stevec = np.abs(2+W1)**(1/W1+0.5)*W2**(-0.5)
    imenovalec = np.abs(2-W1)**(1/W1-0.5)*W2**0.5
    return np.abs(np.log(stevec/imenovalec))
    
wji = np.linspace(3.8,3.99,1000)
plt.plot(4-wji,[lok(w) for w in wji])
plt.xscale("log")
plt.yscale("log")
x = np.log10(4-wji)
y = np.log10([lok(w) for w in wji])
print(np.polyfit(x,y,1))
"""

