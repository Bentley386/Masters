import matplotlib.pyplot as plt
import os
import numpy as np
import re
from matplotlib import animation
from scipy.optimize import curve_fit
from numpy.fft import fft, ifft


def periodic_corr(x,y):
    rez = ifft(fft(x)*fft(y).conj()).real
    return rez/rez[0]

#files = [i for i in os.listdir("./EksitacijeVecjiSampleRate")]
files = [i for i in os.listdir("./EksitacijeSproti")]
N = 1000
pattern1 = re.compile(r"\[[^[]*\]")
#pattern = re.compile(r"\(\d+\.\d+[+-]")

def getEverything(foldername):
    everything=[]
    files = [i for i in os.listdir("./{}".format(foldername))]
    for file in files: 
        f = open("./{}/".format(foldername) + file)

        #f=open("./EksitacijeVecjiSampleRate/" + file)
        data = re.findall(pattern1,f.read())
        every = np.zeros((len(data),500))*1j

        for j in range(len(data)):
            t = list(map(complex,data[j][1:-1].split()))
            every[j]+=[t[i]+t[i+1] for i in range(0,1000,2)]
        everything.append(every)
        f.close()
    return everything
 
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
    
x = np.arange(500)

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

def ExcitationsDuring(every):
    fig, axi = plt.subplots(2,2,sharex=True,sharey=True)
    axi[0][0].bar(x,every[0])
    axi[0][0].set_title("t=0")
    axi[0][0].set_ylabel(r"$N_{ex}$")
    axi[0][1].bar(x,every[20])
    axi[0][1].set_title("t=20")
    axi[1][0].bar(x,every[40])
    axi[1][0].set_title("t=40")
    axi[1][0].set_ylabel(r"$N_{ex}$")
    axi[1][0].set_xlabel(r"$x$")
    axi[1][1].bar(x,every[60])
    axi[1][1].set_title("t=80")
    axi[1][1].set_xlabel(r"$x$")
    plt.savefig("HistogramiLog.pdf")

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
    times = np.linspace(10,1000,50)
    AvgIPR = np.zeros(50)
    for every in everything:
        IPRs = []
        for i in range(50):
            probs = every[i]/sum(every[i])
            IPRs.append(1/sum([v*v for v in probs]))
        AvgIPR+=np.real(np.array(IPRs))
    plt.plot(times,AvgIPR/99)
    plt.yscale("log")
    plt.xscale("log")
    
    
    
    
eks=[]

for T in [20,200,2000]:
    print(T)
    if T==2000:
        forT2000()
        continue
    everything = getEverything("vecEksFiksenT/T{}".format(str(T)))
    numOfExc(everything,T)
č   
plt.plot([20,200,2000],eks)
plt.yscale("log")
plt.xscale("log")

plt.legend()
plt.xlabel(r"$W$")
plt.ylabel(r"$N_{ex}$")
plt.title("Število eks. od W za več časov quencha")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

