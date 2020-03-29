"""
Display both energy bands and excitations
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import matplotlib.lines as mlines


#0 DO 8 (W) ENERGIJE

#Povprecenje st.eks in IPR (T) 

#filmcek   
#20 najnizjih stanj IPR (t) 

#cetrtek 13:00



N = 1000
pattern1 = re.compile(r"\[[^[]*\]")
#pattern = re.compile(r"\(\d+\.\d+[+-]")
#10158825

every = np.zeros((200,99*500))
j=0

def energyPlots():
    for file in ["8166247.txt"]: 
        f = open("../rezultatiN1000Energije/" + file).read()
        data = re.findall(pattern1,f)
        for j in range(int(N/2)):
            energies = []
            for i in range(len(data)):
                energies.append(float(data[i][1:-1].split()[int(N/2)+j]))
                #energies.append(float(data[i][1:-1].split()[int(N/2)+1]))
            if np.amin(energies) > 0.5:
                continue
            plt.plot(np.linspace(3.5,4.5,len(data)),energies)
            plt.yscale("log")
            #energies = np.array([float(s) for s in data[i][1:-1].split()])[int(N/2):]
            #every[i][j:j+500] = energies
        j+=500

energyPlots()
l
def energyAndExcitations(file):
    cmap = plt.get_cmap("binary")
    with open("./energijeDoMaloVecW/" + file, "r") as f:
        energydata = re.findall(pattern1,f.read())
        
    with open("./eksByStanje/" + file, "r") as f:
        excdata = re.findall(pattern1,f.read())
    x = np.linspace(3,5,len(energydata))
    for j in range(int(N/2)):
        energies = []
        for i in range(200):
            energies.append(float(energydata[i][1:-1].split()[int(N/2)+j]))
        if np.amin(energies) < 0:
            continue
        
        excitations = []
        for i in range(200):
            excitations.append(float(excdata[i][1:-1].split(", ")[j]))
        #plt.plot(x,energies)
        #scatter 100 ekv marker 10
        plt.scatter(x[::],energies[::],c=[cmap(e) for e in excitations][::])
    sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    markers=[mlines.Line2D([],[],ls='None',marker="o",c=cmap(s)) for s in sizes]
    plt.legend(markers,list(map(str,sizes)),loc="best")
    #plt.yscale("log") 
#energyAndExcitations("436409.txt")  
energyAndExcitations("10158825.txt")  
#energyPlots()

def lorentz(x,x0):
    HWHM = 0.5
    return np.exp(-(x-x0)**2/(HWHM**2))
    return 1/np.pi * HWHM / ((x-x0)**2 +HWHM**2) 
def density(x,states):
    suma=0
    for r in states:
        suma += lorentz(x,r)
    return suma

def DOS():
    for i in range(0,100,10):
        energies = every[i]
        energies2 = []
        for en in energies:
            if en>10 and en<0.1:
                energies2.append(np.log10(en))
        Es = np.linspace(np.amin(energies2),np.amax(energies2),100)
        dos = [density(E,energies2) for E in Es]
        plt.plot(Es,dos,label=i)



def enCount():
    count = []
    for i in every:
        count.append(sum([1 if (energ<10**(-6) and energ>10**(-7)) else 0 for energ in i]))
    plt.plot(np.arange(100),count)

    plt.legend()    
