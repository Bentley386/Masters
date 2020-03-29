# -*- coding: utf-8 -*-
"""
Everything regarding excitations during quenching past the phase transition point
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import re
from scipy.optimize import curve_fit

files = [i for i in os.listdir("../EksitacijeSproti")]

def getEnergies(T):
    allenergies=[]
    pattern = re.compile(r"\[[^[]*\]")
    folderpath = "../ExcByEnergy/energije/{}".format(T)
    folder = [i for i in os.listdir(folderpath)]
    
    for textfile in folder:
        energies=[]
        with open("{}/{}".format(folderpath,textfile),"r")  as f:
            data = re.findall(pattern,f.read())
        for i in data:
            energies.append(list(map(float,i[1:-1].split())))
        allenergies.append(energies)
    return allenergies

def getExcitations(T):
    allexcitations=[]
    pattern = re.compile(r"\[[^[]*\]")
    folderpath = "../ExcByEnergy/eksitacije/{}".format(T)
    folder = [i for i in os.listdir(folderpath)]
    
    for textfile in folder:
        excitations=[]
        with open("{}/{}".format(folderpath,textfile),"r")  as f:
            data = re.findall(pattern,f.read())
        for i in data:
            excitations.append(list(map(float,i[1:-1].split(","))))
        allexcitations.append(excitations)
    return allexcitations


def excitationsDuringQuench(Ws,times):
    for T in times:
        allExcitations = getExcitations(T)
        averaged = np.zeros(len(Ws))
        for i in allExcitations:
            for j in range(len(i)):
                averaged[j]+=np.sum(i[j])
        averaged/= len(allExcitations)
        
        Ws = np.array(Ws)
        #averaged/=averaged[-1]
        plt.plot(Ws,averaged,label=r"$T={}$".format(T))
        #plt.plot((Ws-4)*T**(0.15)+4,averaged,label=r"$T={}$".format(T))
    plt.xlabel(r"$W$")
    plt.ylabel(r"$N_{exc}$")
    plt.legend(loc="best")
    plt.grid()
    
def model(x,A,e1,e2):
    return A/x**e1/ np.log(x)**e2    

def linRegress(T,N):
    A = np.ones((len(T),3))
    A[:,1] = -np.log(T)
    A[:,2] = -np.log(np.log(T))
    b = np.log(N)    
    return np.linalg.lstsq(A,b,rcond=None)[0]

def exponentForFinal(times,divide=1,fit=False):
    averaged = np.zeros(len(times))
    for i in range(len(times)):
        T=times[i]
        allExcitations = getExcitations(T)
        for j in allExcitations:
            averaged[i]+=np.sum(j[-1])
    averaged /= len(allExcitations)
    plt.plot(np.array(times)/divide,averaged)
    plt.xscale("log")
    plt.yscale("log")
    if fit:
        params=curve_fit(model,np.array(times)/divide,averaged,(1,0,2))[0]
        params2=linRegress(times,averaged)
        casi = np.linspace(times[0],times[-1],1000)
        print(params[1])
        print(params[2])
        print(params2[1])
        print(params2[2])
        plt.plot(casi,[model(x,params[0],params[1],params[2]) for x in casi])
        plt.plot(casi,[np.exp(params2[0])/x**params2[1]/np.log(x)**params2[2] for x in casi],"--")
        
        
def contourExcByEnergy(Ws,time):
    #partition=26
    partition = 1000
    energyRange = 10**np.linspace(-13,0,partition)
    excitations = np.zeros((partition,len(Ws)))
    number = np.zeros((partition,len(Ws)))
    allExcitations = getExcitations(time)
    allEnergies = getEnergies(time)
    for i in range(len(allEnergies)):
        for w in range(len(Ws)):
            for j in range(500):
                energy = allEnergies[i][w][j]
                if energy>1:
                    continue
                where=0
                while where<partition:
                    if energy < energyRange[where]:
                        excitations[where][w] += allExcitations[i][w][j]
                        number[where][w] += 1
                        break
                    where += 1
    for w in range(len(Ws)):
        for where in range(partition):
            if number[where][w] > 0:
                excitations[where][w] /= number[where][w]
                
    #plt.xlabel(r"$W$")
    #plt.ylabel(r"$E$")
    #plt.yscale("log")
    plt.plot(energyRange,excitations[:,-1],label=str(time))
    #plt.pcolormesh(excitations,vmax=0.6)
    #plt.contourf(Ws,energyRange,excitations,levels=np.linspace(np.amin(excitations),np.amax(excitations),50))
    #plt.colorbar()

def EexcMax(times):
    emax = []
    partition=1000
    energyRange = 10**np.linspace(-13,0,partition)
    for time in times:
        excitations = np.zeros(partition)
        number = np.zeros(partition)
        allExcitations = getExcitations(time)
        allEnergies = getEnergies(time)
        for i in range(len(allEnergies)):
            for j in range(500):
                energy = allEnergies[i][10][j]
                if energy >= 1:
                    continue
                where = 0
                while where<partition:
                    if energy < energyRange[where]:
                        excitations[where] += allExcitations[i][10][j]
                        number[where] += 1
                        break
                    where += 1
        for i in range(1,partition+1):
            if number[-i] > 0:
                excitations[-i] /= number[-i]
                if excitations[-i] >= 0.25:
                    print(energyRange[-i])
                    emax.append(energyRange[-i])
                    break
    plt.plot(times,emax)
    plt.yscale("log")
    plt.xscale("log")
    
#EexcMax([200,600,2000,6000,20000,60000,200000])
#EexcMax([400,900,4000,9000,40000])
       

#excitationsDuringQuench([3.5+0.1*i for i in range(1,11)],[10,100,1000,10000,30,300,3000])
#excitationsDuringQuench([3+0.1*i for i in range(1,21)],[20,200,2000,20000,60,600,6000])
#excitationsDuringQuench([2+0.1*i for i in range(1,41)],[40,400,4000,40000,90,900,9000])
 
#excitationsDuringQuench([3+0.1*i for i in range(1,21)],[0])
#excitationsDuringQuench([2+0.1*i for i in range(1,41)],[40000])    
    
#excitationsDuringQuench([2+0.1*i for i in range(1,41)],[400,4000,40000,900,9000])

    
#exponentForFinal([10,30,100,300,1000,3000,10000])
#exponentForFinal([600,2000,6000,20000,60000,200000],fit=True)
#exponentForFinal([400,900,4000,9000,40000],fit=True)
            
        
        
 
#contourExcByEnergy([3.5+0.1*i for i in range(1,11)],10000)
#contourExcByEnergy([3+0.1*i for i in range(1,21)],2000)
    
#for times in [200,600,2000,6000,20000]:
#    contourExcByEnergy([3+0.1*i for i in range(1,21)], times)
#plt.legend()

#8166247 2525443
def singleState(stanje):
    N=1000
    path = "../poStanjih/"
    pattern = re.compile(r"\[[^[]*\]")
    with open(path + "energije/8166247.txt","r") as f:
        energies = re.findall(pattern,f.read())
    with open(path + "eksitacije/{}/8166247.txt".format(str(stanje)),"r") as f:
        transitions = re.findall(pattern,f.read())
    
    energies = [list(map(float,e[1:-1].split()))[int(N/2):] for e in energies]
    transitions = [list(map(float,e[1:-1].split(",")))[int(N/2):] for e in transitions]
    transitions = np.array(transitions)
    #print([i for i in transitions[15] if i>0.01])
    z = transitions.flatten()
    y = np.array(energies).flatten()
    Wji = [3+0.1*i for i in range(1,21)]
    x = np.repeat(Wji,int(N/2))
    cmap = plt.get_cmap("jet")
    plt.yscale("log")
    #plt.scatter(x,y,c=z,cmap=cmap,s=z*30)
    plt.xlabel(r"$W$")
    plt.ylabel(r"$E$")
    plt.title("Seed: 8166247, Stanje z {}. najvišjo negativno energijo".format(stanje))
    for i in range(len(z)):
        if np.abs(z[i]) > 0.1:
            plt.scatter([x[i]],[y[i]],c=[z[i]],cmap=cmap,s=[z[i]*30])
            plt.text(x[i],y[i],str(i%(int(N/2))+1))
    plt.colorbar()
    #plt.savefig("{}.pdf".format(stanje))
    #plt.clf()
#    cnt = plt.tricontourf(x,y,z,levels=np.linspace(np.amin(z),np.amax(z),50))
#    for c in cnt.collections: #remove ugly  white linesle
#        c.set_edgecolor("face")
#    plt.colorbar(cnt)    

singleState(1)

def singleStateValence(stanje):
    N=1000
    path = "../poStanjih/"
    pattern = re.compile(r"\[[^[]*\]")
    with open(path + "energije/8166247.txt","r") as f:
        energies = re.findall(pattern,f.read())
    with open(path + "eksitacije/{}/8166247.txt".format(str(stanje)),"r") as f:
        transitions = re.findall(pattern,f.read())
    
    energiesCond = [list(map(float,e[1:-1].split()))[int(N/2):] for e in energies]
    transitionsCond = [list(map(float,e[1:-1].split(",")))[int(N/2):] for e in transitions]
    transitionsCond = np.array(transitionsCond)
    transitionsVal = [list(map(float,e[1:-1].split(",")))[:int(N/2)][::-1] for e in transitions]
    transitionsVal = np.array(transitionsVal)
    z = transitionsCond.flatten()
    y = np.array(energiesCond).flatten()
    Wji = [3+0.1*i for i in range(1,21)]
    x = np.repeat(Wji,int(N/2))
    cmap = plt.get_cmap("jet")
    fig, [ax1, ax2] = plt.subplots(2,figsize=(12,12))
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    plot = ax1.scatter(x,y,c=z,cmap=cmap,s=z*30)
    plt.colorbar(plot, ax=ax1)
    #ax1.set_xlabel(r"$W$")
    ax1.set_ylabel(r"$E$")
    ax1.set_title("Seed: 8166247, Stanje z {}. najvišjo negativno energijo, prevodni pas".format(stanje))
    for i in range(len(z)):
        if np.abs(z[i]) > 0.1:
            ax1.text(x[i],y[i],str(i%(int(N/2))+1))

    z = transitionsVal.flatten()
    plot = ax2.scatter(x,y,c=z,cmap=cmap,s=z*30)
    plt.colorbar(plot, ax=ax2)
    ax2.set_xlabel(r"$W$")
    ax2.set_ylabel(r"$-E$")
    ax2.set_title("Seed: 8166247, Stanje z {}. najvišjo negativno energijo, valenčni pas".format(stanje))
    for i in range(len(z)):
        if np.abs(z[i]) > 0.1:
            ax2.text(x[i],y[i],str(i%(int(N/2))+1))
    plt.tight_layout()
    plt.savefig("{}.pdf".format(stanje))
    plt.clf()
#    cnt = plt.tricontourf(x,y,z,levels=np.linspace(np.amin(z),np.amax(z),50))
#    for c in cnt.collections: #remove ugly  white linesle
#        c.set_edgecolor("face")
#    plt.colorbar(cnt) 

for i in range(1,11):
    break
    singleStateValence(i)
        
        
        
        
        
        
        
        