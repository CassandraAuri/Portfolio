from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
import mplcyberpunk
from MFA import MFA
set_token("https://vires.services/ows", set_default=True,
          token="kmxv5mTTyYwzw4kQ9lsCkGfQHtjjJRVZ")  # key

plt.style.use("cyberpunk")  # Dark mode!

fig, axes = plt.subplots(nrows=3,
                         figsize=(10, 6), sharex=True, sharey=False
                         )

time_range = (datetime(2021, 3, 18, 8, 11),
              datetime(2021, 3, 18, 8, 25)) 


labels = ["Swarm A", "Swarm B", "Swarm C"] #Labels of space-craft interested in
measurements = ["B_NEC", "Ehx"] #Measurement names from swarm


collectionE = [
    "SW_EXPT_EFIA_TCT16",
    "SW_EXPT_EFIB_TCT16",
    "SW_EXPT_EFIC_TCT16",
]
collectionB = [
    "SW_OPER_MAGA_HR_1B",
    "SW_OPER_MAGB_HR_1B",
    "SW_OPER_MAGC_HR_1B"
]
collectionF = [
    "SW_OPER_FACATMS_2F",
    "SW_OPER_FACBTMS_2F",
    "SW_OPER_FACCTMS_2F"] #Data packages from swarm


def requester(sc_collection, measurement, **kwargs): #Requests data from swarm
    try:
        request = SwarmRequest()
        request.set_collection(sc_collection)
        request.set_products(measurements=measurement, models="CHAOS")

        data = request.get_between(time_range[0], time_range[1], **kwargs)
        df = data.as_dataframe()
    except:
        df = []
    return df


def graphingB(index, arrayx, arrayy):
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[0].plot(arrayx, arrayy,
                 color=color[index], label=r"$B_Y$" + ", "+labels[index])
    axes[0].legend(loc=2)
    axes[0].set_ylabel(r"$\Delta$  $B_{Y}$ (nT)")


def graphingF(index, arrayx, arrayy):
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[2].plot(arrayx, arrayy,
                 color=color[index], label="FAC, "+labels[index])
    axes[2].legend(loc=2)
    axes[2].set_ylabel("FAI Intensity (kR)")


def graphingE(index, dataset, arrayx, arrayy): 
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[1].plot(arrayx, arrayy,
                 color=color[index], label=dataset+","+labels[index])
    axes[1].set_ylabel(r"$E_{N}$ $(mV/m)$")
    axes[1].legend(loc=2)

def Coordinate_change(lattiude,longitude,radius): #Coordinate change
        a, b, e2 = 6378137.0, 6356752.3142, 0.00669437999014 #From DRS80
        lat, lon, h = lattiude, longitude, radius 
        print(lat)
        v = a/np.sqrt(1-e2*np.sin(lat)*np.sin(lat)) #logic
        x = (v+h)*np.cos(lat)*np.cos(lon)
        y = (v+h)*np.cos(lat)*np.sin(lon)
        z = (v*(1-e2)+h)*np.sin(lat)
        print(x)
        return [x, y, z]

def requesterarraylogic():
    def B():

        for i in range(len(collectionB)): #Goes through every satellite
            # Data package level, data/modes, **kwargs
            ds = requester(collectionB[i], measurements[0],
                           asynchronous=False, show_progress=False)

            def model(): #Gets B field from CHAOS model for Mean-Field
                Bmodel = ds["B_NEC_CHAOS"]
                return Bmodel.to_numpy()
            bmodel = model()
            print(ds["Radius"].to_numpy())
            radius, lattiude, longitude = ds["Radius"].to_numpy(), ds['Latitude'].to_numpy(), ds['Longitude'].to_numpy() #Gets Emphermis data
            Bdata = ds["B_NEC"]  # data package
            # Finds the time which is stored as a row header (ie row name)
            time = Bdata.index
            #time = np.delete(time, -1)
            b = Bdata.to_numpy()
            # since minutes only, could start half way nbetween a measurement
            r_nec=Coordinate_change(lattiude,longitude,radius)
            barranged=np.zeros((len(b),3))
            bmodelarranged=np.zeros((len(b),3))
            for j in range(len(b)): #Re-arranges into proper (n x 3 ) matricies, ugly but works
                for k in range(3):
                    barranged[j][k]=b[j][k]
                    bmodelarranged[j][k]=bmodel[i][k]
            datamfa=MFA(barranged,bmodelarranged, np.asarray(r_nec).T) #Calls MFA with (3xn) vectors
            graphingB(i, time, datamfa[:,1])
            #graphingB(i, time, datamfa[:,2])
            #graphingB(i, time, datamfa[:,0]) #Collects all the compressional B values


    def E():
        lens = len(collectionE)*2
        for i in range(len(collectionE)):
            for j in range(1):
                e = []
                ds = requester(
                    collectionE[i], measurements[j+1], asynchronous=False, show_progress=False)
                if(len(ds) != 0):
                    Eall = ds[measurements[j+1]]
                    time = Eall.index
                    Eall = pd.Series.to_numpy(Eall)

                    graphingE(i, measurements[j+1], time, Eall)

    def F():
        for i in range(len(collectionF)):
            b = []
            ds = requester(
                collectionF[i], "FAC", asynchronous=False, show_progress=False)
            fac = ds["FAC"]
            time = fac.index
            fac = pd.Series.to_numpy(fac)

            graphingF(i, time, fac)

    B()
    E()
    F()


requesterarraylogic()

fig.supxlabel("Time (Day:Hour:Minute)")
fig.suptitle("Time Versus Auroral Parameters From Swarm Spacecraft")

#for i in range(len(axes)):  # final touches
  #mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
# mplcyberpunk.add_gradient_fill(
# ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")
plt.show()
