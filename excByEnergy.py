# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:32:48 2020

@author: Admin
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import re
from scipy.optimize import curve_fit

files = [i for i in os.listdir("./EksitacijeSproti")]

def getEnergies(T):
    allenergies=[]
    pattern = re.compile(r"\[[^[]*\]")
    folderpath = "./ExcByEnergy/energije/{}".format(T)
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
    folderpath = "./ExcByEnergy/eksitacije/{}".format(T)
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
    #plt.plot(np.array(times)/divide,averaged)
    plt.xscale("log")
    plt.yscale("log")
    if fit:
        params=curve_fit(model,np.array(times)/divide,averaged,(1,0,2))[0]
        params2=linRegress(times,averaged)
        casi = np.linspace(times[0],times[-1],1000)
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
                energy = allEnergies[i][-1][j]
                if energy >= 1:
                    continue
                where = 0
                while where<partition:
                    if energy < energyRange[where]:
                        excitations[where] += allExcitations[i][-1][j]
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
    
#EexcMax([200,600,2000,6000,20000])
#EexcMax([400,900,4000,9000,40000])
       

#excitationsDuringQuench([3.5+0.1*i for i in range(1,11)],[10,100,1000,10000,30,300,3000])
#excitationsDuringQuench([3+0.1*i for i in range(1,21)],[20,200,2000,20000,60,600,6000])
#excitationsDuringQuench([2+0.1*i for i in range(1,41)],[40,400,4000,40000,90,900,9000])
 
#excitationsDuringQuench([3+0.1*i for i in range(1,21)],[20000])
#excitationsDuringQuench([2+0.1*i for i in range(1,41)],[40000])    
    
#excitationsDuringQuench([2+0.1*i for i in range(1,41)],[400,4000,40000,900,9000])

    
#exponentForFinal([10,30,100,300,1000,3000,10000])
#exponentForFinal([600,2000,6000,20000],fit=True)
#exponentForFinal([400,900,4000,9000,40000],fit=True)
            
        
        
 
#contourExcByEnergy([3.5+0.1*i for i in range(1,11)],10000)
#contourExcByEnergy([3+0.1*i for i in range(1,21)],2000)
    
#for times in [200,600,2000,6000,20000]:
#    contourExcByEnergy([3+0.1*i for i in range(1,21)], times)
#plt.legend()

     
        
        
        
        
        
        
        
        