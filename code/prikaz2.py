"""
Display both energy bands and excitations
"""
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import scipy.integrate
import re
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
from scipy.optimize import curve_fit

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

def energyPlotsOld():
    for file in ["8166247.txt"]: 
        f = open("../Missing/" + file).read()
        data = re.findall(pattern1,f)
        for j in range(int(N/2)):
            energies = []
            for i in range(len(data)):
                energies.append(float(data[i][1:-1].split()[int(N/2)+j]))
                #energies.append(float(data[i][1:-1].split()[int(N/2)+1]))
            if np.amin(energies) > 0.0005:
                continue
            plt.plot(np.linspace(3,5,len(data)),energies)
            plt.yscale("log")
            #energies = np.array([float(s) for s in data[i][1:-1].split()])[int(N/2):]
            #every[i][j:j+500] = energies
        j+=500

#energyPlotsOld()

def energyPlots():
    fig = plt.figure(figsize=(12,12))
    gs= gridspec.GridSpec(ncols=3, nrows=3, figure=fig)    
    ax = fig.add_subplot(gs[:2, :])
    ax2 = ax.twinx()
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[2, 2])
    axins = zoomed_inset_axes(ax, 10, loc=7) # zoom-factor: 2.5, location: upper-left
    
    for file in ["436409.txt"]: 
        f = open("../EnergijeDoVecW/" + file).read()
        data = re.findall(pattern1,f)
        for j in range(int(N/2)):
            energies = []
            for i in range(len(data)):
                energies.append(float(data[i][1:-1].split()[int(N/2)+j]))
                #energies.append(float(data[i][1:-1].split()[int(N/2)+1]))
            if np.amin(energies) > 0.05:
                continue
            ax.plot(np.linspace(2,6,len(data)),energies)
            axins.plot(np.linspace(2,6,len(data)), energies)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    axins.set_xlim(4.24, 4.28)
    axins.set_ylim(4.815e-05,0.0003)
    ax.set_yscale("log")
    ax.set_xlabel(r"$W$")
    ax.set_ylabel(r"$E$")
    ax.set_title("Energijski nivoji skozi prehod med fazama")
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    #plt.tight_layout()
    aux = []
    for i in [100,300,400]:
        en = np.array(list(map(float,data[i][1:-1].split())))
        aux.append([(np.linspace(2,6,800)[i],en[int(N/2)]),en])
            #energies = np.array([float(s) for s in data[i][1:-1].split()])[int(N/2):]
            #every[i][j:j+500] = energies
        #j+=500
    return [fig, ax, ax2, ax3, ax4, ax5,aux]
        
def energyAndExcitations(file):
    cmap = plt.get_cmap("binary")
    with open("../energijeDoMaloVecW/" + file, "r") as f:
        energydata = re.findall(pattern1,f.read())
        
    with open("../eksByStanje/" + file, "r") as f:  #T=200
        excdata = re.findall(pattern1,f.read())
    x = np.linspace(3,5,len(energydata))
    for j in range(int(N/2)):
        energies = []
        for i in range(200):
            energies.append(float(energydata[i][1:-1].split()[int(N/2)+j]))
        if np.amin(energies) < 0 or np.amin(energies) > 0.001:
            continue
        
        excitations = []
        for i in range(200):
            excitations.append(float(excdata[i][1:-1].split(", ")[j]))
        plt.plot(x,energies)
        #scatter 100 ekv marker 10
        plt.scatter(x[::10],energies[::10],c=[cmap(e) for e in excitations][::10],s=[15*e for e in excitations][::10],zorder=10000)
    sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    markers=[mlines.Line2D([],[],ls='None',marker="o",c=cmap(s)) for s in sizes]
    plt.legend(markers,list(map(str,sizes)),loc="best")
    plt.yscale("log") 
    plt.xlabel(r"$W$")
    plt.ylabel(r"$E$")
    plt.title(r"Eksitacije tekom quencha, $v=0.01, W \in [3,5]$")
    plt.grid()
#energyAndExcitations("436409.txt")  
#energyAndExcitations("10158825.txt") 
#plt.savefig("Figures/EksTekom2.pdf") 
#energyPlots()

def lorentz(x,x0):
    HWHM = 0.01
    return 1/(HWHM*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(HWHM**2)/2)
    #return 1/np.pi * HWHM / ((x-x0)**2 +HWHM**2) 
    
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

def dosmodel(x,A,B,C):
    return A/x**B/np.log(x)**C

def lengthEmpiric(vs,ws):
    """loc. length by averaging"""
    ms = vs[1::2]
    ts = np.concatenate((ws[2::2],np.array([ws[-1]])))  #notation in article
    n = len(ms)
    lamb = np.sum(np.log(np.abs(ts))-np.log(np.abs(ms)))/n
    return 1/np.abs(lamb)

def forMasters():
    """Generate pic which I will finally put in masters"""
    fig, ax, ax2, ax3, ax4 , ax5, aux = energyPlots()
    dosaxi = [ax3,ax4,ax5]
    ax.grid(True)
    np.random.seed(436409)
    omega1 = np.random.rand(1000)-0.5
    omega2 = np.random.rand(1000)-0.5
    length = []
    for W in np.linspace(2,6,800):
        vs = W*omega2
        ws = np.ones(1000)+0.5*W*omega1
        length.append(lengthEmpiric(vs,ws))
    ax2.plot(np.linspace(2,6,800),length,ls="--",alpha=0.5,color="k")
    ax2.set_ylabel(r"$\Lambda$",rotation=270)

    ax3.set_ylabel(r"$g$")    
    color = plt.get_cmap("copper")(0.5)
    for i in range(3):
        dosaxi[i].set_xlabel(r"$E$")
        Es = np.linspace(10**(-15),3,1000)
        dos = [density(E,aux[i][1]) for E in Es]
        y = np.array(dos)
        y/= scipy.integrate.trapz(y,Es)
        dosaxi[i].plot(Es,y,color="blue")
        if i==2:
            Es = 10**np.linspace(-7,-9,100)
            dos = [density(E,aux[i][1]) for E in Es]
            y = np.array(dos)
            A,B,C = curve_fit(dosmodel,Es,y)
            print(A,B,C)
            plt.clf()
            plt.xscale("log")
            plt.yscale("log")
            plt.plot(Es,y)
            č
        
        dosaxi[i].plot(-Es,y,color="blue")
        
        ax.axvline(x=aux[i][0][0],color=color,ls=":")
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(dosaxi[i].transData.transform([0,dosaxi[i].get_ylim()[1]-dosaxi[i].get_ylim()[1]/8]))
        coord2 = transFigure.transform(ax.transData.transform([aux[i][0][0],10**(-12)]))
        line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),transform=fig.transFigure,color=color)
        fig.lines.append(line)
        dosaxi[i].grid(True)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("Figures/Nicegraph.pdf")
#forMasters()

def enCount():
    count = []
    for i in every:
        count.append(sum([1 if (energ<10**(-6) and energ>10**(-7)) else 0 for energ in i]))
    plt.plot(np.arange(100),count)

    plt.legend()    
