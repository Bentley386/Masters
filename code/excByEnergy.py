# -*- coding: utf-8 -*-
"""
Everything regarding excitations during quenching past the phase transition point
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import re
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation

sirinca=434.62277*0.0138
visinca=318.777*0.0138
plt.rc('text', usetex=True)
plt.rc('font', size=12)
#plt.rc('figure', figsize=(sirinca, visinca))

files = [i for i in os.listdir("../../EksitacijeSproti")]

def getEnergies(T):
    allenergies=[]
    pattern = re.compile(r"\[[^[]*\]")
    folderpath = "../../ExcByEnergy/energije/{}".format(T)
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
    folderpath = "../../ExcByEnergy/eksitacije/{}".format(T)
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

def model2(x,A,B):
    return A*x**B

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
    #partition=50
    partition = 20
    energyRange = 10**np.linspace(-13,-4,partition)
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
                
    plt.xlabel(r"$W$",fontsize=15)
    plt.ylabel(r"$E$",fontsize=15)
    #plt.title(r"Eksitacije na interval energije, 50 binov. $v=0.0001$")
    #plt.yscale("log")
    #plt.plot(energyRange,excitations[:,-1],label=str(time))
    #plt.pcolormesh(excitations,vmax=0.6)
    cnt = plt.contourf(Ws,energyRange,excitations,levels=np.linspace(np.amin(excitations),np.amax(excitations),50),cmap="plasma")
    for c in cnt.collections: #remove ugly  white lines
        c.set_edgecolor("face")
    plt.colorbar()
    plt.tight_layout()

#contourExcByEnergy([3+0.1*i for i in range(1,21)],2000)
#plt.savefig("EksBini1.pdf")

def EexcMaxCusBins(times,findBins=False,bins=[]):
    emax = []
    if findBins:
        partition=200
        energyRange = 10**np.linspace(-13,0,partition)
    else:
        partition = len(bins)
        energyRange = bins
    
    ForBin=[]
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
                if findBins:
                    ForBin.append(energy)
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
                    emax.append(energyRange[-i])
                    break
        else:
            print("else")
            emax.append(energyRange[0])
    if findBins:
        return np.histogram_bin_edges(ForBin,"stone")
    
    plt.plot(2/np.array(times),emax,":",marker=".")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$v$")
    plt.ylabel(r"$E_{max}$")
    plt.title(r"Energije pri katerih padejo eksitacije na $0.25$")
    plt.grid()
#    
def EexcMax(times,partition,color,marker,indeks):
    emax = []
    energyRange = np.linspace(1e-13,1,partition)
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
                    emax.append(energyRange[-i])
                    break
    
    x = 2/np.array(times)
    y = emax
    A,B = curve_fit(model2,x,y)[0]
    xx = np.linspace(x[0],x[-1],100)
    plt.plot(xx,[model2(i,A,B) for i in xx],"--",color=color)
    #plt.text(x[1+i],y[1+i],r"$E \propto v^{%.3f}$" % (B))
    plt.plot(x,y,color=color,marker=marker,lw=0,label=f"Particija: {partition}, " + r"$E \propto v^{%.3f}$" % (B))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$v$")
    plt.ylabel(r"$E_{max}$")
    #plt.title(r"Energije pri katerih padejo eksitacije na $0.25, W: 3 \to 5$")
    plt.grid()    

#bins = EexcMaxCusBins([600,2000,6000,20000,60000,200000],True)
#bins = np.sort(np.concatenate((10**np.linspace(-4,-3,300)
#,bins)))
#print("hm")
#EexcMaxCusBins([600,2000,6000,20000,60000,200000],bins=bins)
#colors=["r","c","g","m"]
#markers=["x"]*4
#for i in range(4):
#    EexcMax([600,2000,6000,20000,60000,200000],200+i*100,colors[i],markers[i],i)
#plt.legend(loc="best")
#plt.savefig("Emax.pdf")
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
    
#for times in [200,600,2000,6000,20000]:
#    contourExcByEnergy([3+0.1*i for i in range(1,21)], times)
#plt.legend()

#8166247 2525443
def singleState(stanje):
    N=1000
    path = "../../poStanjih/"
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
    plt.xlabel(r"$W$",fontsize=15)
    plt.ylabel(r"$E$",fontsize=15)
    #plt.title("Prehodi iz quenchanega stanja z najvišjo negativno energijo, prevodni pas".format(stanje))
    for i in range(len(z)):
        if np.abs(z[i]) > 0.1:
            plt.scatter([x[i]],[y[i]],c=[z[i]],cmap=cmap,s=[z[i]*30],zorder=10000)
            if(i%(int(N/2))+1>=5):
                plt.text(x[i],y[i]+0.0005,str(i%(int(N/2))+1))
            else:
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
    path = "../../poStanjih/"
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
    fig, [ax1, ax2] = plt.subplots(2)
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    plot = ax1.scatter(x,y,c=z,cmap=cmap,s=z*30)
    plt.colorbar(plot, ax=ax1)
    #ax1.set_xlabel(r"$W$")
    ax1.set_ylabel(r"$E$")
    #ax1.set_title("Prehodi iz quenchanega stanja z najvišjo negativno energijo, prevodni pas".format(stanje))
    for i in range(len(z)):
        if np.abs(z[i]) > 0.1:
            if(i%(int(N/2))+1>=5):
                ax1.text(x[i],y[i]+0.0005,str(i%(int(N/2))+1))
            else:
                ax1.text(x[i],y[i],str(i%(int(N/2))+1))
            
    z = transitionsVal.flatten()
    plot = ax2.scatter(x,y,c=z,cmap=cmap,s=z*30)
    plt.colorbar(plot, ax=ax2)
    ax2.set_xlabel(r"$W$")
    ax2.set_ylabel(r"$-E$")
    #ax2.set_title("Prehodi iz quenchanega stanja z najvišjo negativno energijo, valenčni pas".format(stanje))
    ax2.invert_yaxis()
    for i in range(len(z)):
        if np.abs(z[i]) > 0.1:
            if (i%(int(N/2))+1>=5):
                ax2.text(x[i],y[i]+0.0005,str(i%(int(N/2))+1))
            else:
                ax2.text(x[i],y[i],str(i%(int(N/2))+1))
    plt.tight_layout()
    plt.savefig("{}.pdf".format(stanje))
    #plt.clf()
    
#singleStateValence(1)
    #cnt = plt.tricontourf(x,y,z,levels=np.linspace(np.amin(z),np.amax(z),50))
    #for c in cnt.collections: #remove ugly  white linesle
    #    c.set_edgecolor("face")
    #plt.colorbar(cnt) 

#for i in range(1,11):
#    singleStateValence(i)
#    break
#        
#        
        
        
    
def animacija():
    pattern = re.compile(r"\[[^[]*\]")
    folderpath = "../Animacija"
    quenchana = []
    prvaval = []
    drugaval = []
    with open("{}/quenchana.txt".format(folderpath),"r")  as f:
        data = re.findall(pattern,f.read())
        for i in data:
            quenchana.append(list(map(complex,i[1:-1].split())))
    with open("{}/prvaVal.txt".format(folderpath),"r")  as f:
        data = re.findall(pattern,f.read())
        for i in data:
            prvaval.append(list(map(complex,i[1:-1].split())))        
    with open("{}/drugaVal.txt".format(folderpath),"r")  as f:
        data = re.findall(pattern,f.read())
        for i in data:
            drugaval.append(list(map(complex,i[1:-1].split()))) 
            
    fig, [ax1,ax2,ax3] = plt.subplots(3,sharex=True)
    x = np.arange(1000)
    def animiraj(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.plot(x,np.abs(quenchana[frame])**2)
        ax2.plot(x,np.abs(prvaval[frame])**2)
        ax3.plot(x,np.abs(drugaval[frame])**2)
        ax1.set_title("Quenchana prva")
        ax2.set_title("Lastna prva")
        ax3.set_title("Lastna druga")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Seed: 8166247, W={}".format(str(3+frame*0.01)))

    ani = FuncAnimation(fig,animiraj,range(200),interval=500)
    ani.save("anim.mp4")
#animacija()