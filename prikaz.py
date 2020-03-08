import matplotlib.pyplot as plt
import os
import numpy as np
import re

def numOfExcitationsReverse():
    files = [i for i in os.listdir("./rezultatiN1000Tau1Obratno")]
    N = len(files)
    pattern = re.compile(r"\(\d+\.\d+[+-]")
    Ts = np.linspace(10,1000,50)
    runningAvg = np.zeros(50)
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i) for i in np.linspace(0,1,N)]
    for file in files:
        f = open("./rezultatiN1000Tau1Obratno/" + file)
        data = np.array([float(s[1:-1]) for s in re.findall(pattern,f.readlines()[0])])
        runningAvg+=data
        #plt.plot(Tji,data,color=colors.pop())
        f.close()
    plt.plot(Ts,runningAvg/N,label="V levo")

def numOfExcitations():
    files = [i for i in os.listdir("./rezultatiN1000Tau1")]
    N = len(files)
    pattern = re.compile(r"\(\d+\.\d+[+-]")
    Ts = 2*np.linspace(10,1000,50)
    runningAvg = np.zeros(50)
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i) for i in np.linspace(0,1,N)]
    for file in files:
        f = open("./rezultatiN1000Tau1/" + file)
        data = np.array([float(s[1:-1]) for s in re.findall(pattern,f.readlines()[0])])
        runningAvg+=data
        #plt.plot(Tji,data,color=colors.pop())
        f.close()
    plt.plot(Ts,runningAvg/(N),label="V desno")
 
    
numOfExcitations()    
    
plt.yscale("log")
plt.legend()
plt.xscale("log")

č
plt.xlabel(r"$T$")
plt.ylabel(r"$N_{ex}$")
plt.title("Št. eks. v odv. od dolžine quencha. $\delta t = 1, W \in [3.5,4.5], m=0$")
plt.grid(True)
#plt.savefig("N2000Tau1.pdf")
#plt.savefig("N1000Tau1ObratnoOboje.pdf")