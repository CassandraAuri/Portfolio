from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
import mplcyberpunk
from MFA import MFA
import navpy
set_token("https://vires.services/ows", set_default=True,
          token="kmxv5mTTyYwzw4kQ9lsCkGfQHtjjJRVZ")  # key

plt.style.use("cyberpunk")  # Dark mode!

fig, axes = plt.subplots(nrows=3,
                         figsize=(10, 6), sharex=True, sharey=False
                         )

time_range = (datetime(2021, 3, 18, 8, 11),
              datetime(2021, 3, 18, 8, 20))
minutes = int((time_range[1] - time_range[0]
               ).total_seconds() / 60)  # 3 second␣
cadencefield = 100  # Polling rate for location for EBdata

labels = ["Swarm A", "Swarm B", "Swarm C"]
measurements = ["B_NEC", "Ehx"]


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
    "SW_OPER_FACCTMS_2F"]


def requester(sc_collection, measurement, **kwargs):
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


def Convert_to_MFA(lattiude, longitude, radius, data, length):  # location vector, data vector,
    def Coordinate_change():
        a, b, e2 = 6378137.0, 6356752.3142, 0.00669437999014
        lat, lon, h = lattiude, longitude, radius
        v = a/np.sqrt(1-e2*np.sin(lat)*np.sin(lat))
        x = (v+h)*np.cos(lat)*np.cos(lon)
        y = (v+h)*np.cos(lat)*np.sin(lon)
        z = (v*(1-e2)+h)*np.sin(lat)
        return [x, y, z]

    locationNEC = Coordinate_change()
    datamfa = []
    for i in range(length):
        dataselected = np.zeros((cadencefield, 3))
        locationselected = np.zeros((cadencefield, 3))
        for j in range(cadencefield):  # index [(i+1)*j]
            dataselected[j] = data[(i+1)*j]
            for k in range(3):
                locationselected[k] = locationNEC[k][(i+1)*j]
        # Gives the average of the column values
        xmean = np.average(dataselected, axis=0)
        datamfa.append(MFA(dataselected, xmean, locationselected))
    return datamfa


def requesterarraylogic():
    def B():

        for i in range(len(collectionB)):
            bmodel = []
            b = []
            time = []
            # Data package level, data/modes, **kwargs
            ds = requester(collectionB[i], measurements[0],
                           asynchronous=False, show_progress=False)

            def model():
                Bmodel = ds["B_NEC_CHAOS"]
                time_model = Bmodel.index
                Bmodel = pd.Series.to_numpy(Bmodel)  # turns to numpy array
                # for i in range(len(Ball))
                for j in range(len(Bmodel[:])):
                    # flattens array for the Northward data ie 2nd index
                    bmodel.append(Bmodel[:][j][1])
                return bmodel
            #bmodel = model()
            radius, lattiude, longitude = ds["Radius"].array, ds['Latitude'].array, ds['Longitude'].array
            Bdata = ds["B_NEC"]  # data package
            # Finds the time which is stored as a row header (ie row name)
            time = Bdata.index
            #time = np.delete(time, -1)
            b = Bdata.array
            # since minutes only, could start half way nbetween a measurement

            while(time.size/cadencefield != float(int(time.size/cadencefield))):
                time = time.delete(-1)
                b = np.delete(b, -1, 0)
                lattiude = np.delete(lattiude, -1, 0)
                longitude = np.delete(longitude, -1, 0)
                radius = np.delete(radius, -1, 0)
            length = int(time.size/cadencefield)
            datamfa = Convert_to_MFA(
                lattiude, longitude, radius, b, length)
            datamfa = np.reshape(datamfa, (3, length, cadencefield))
            print(np.shape(datamfa))
            #graphingB(i, time, np.reshape(datamfa[0], -1))
            graphingB(i, time, np.reshape(datamfa[1], -1))
            #graphingB(i, time, np.reshape(datamfa[2], -1))

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

# for i in range(len(axes)):  # final touches
# mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
# mplcyberpunk.add_gradient_fill(
# ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")
plt.show()
