from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from tqdm.notebook import tqdm
import mplcyberpunk
set_token("https://vires.services/ows", set_default=True,
          token="kmxv5mTTyYwzw4kQ9lsCkGfQHtjjJRVZ")  # key

plt.style.use("cyberpunk")  # Dark mode!

fig, axes = plt.subplots(nrows=3,
                         figsize=(10, 6), sharex=True, sharey=False
                         )

start_time = dt.datetime(2021, 3, 18, 8, 13)
end_time = dt.datetime(2021, 3, 18, 8, 22)  # start and end time

labels = ["Swarm A", "Swarm B", "Swarm C"]
measurements = ["B_NEC", "Ehx"]


collectionE = [
    "SW_EXPT_EFIA_TCT02",
    "SW_EXPT_EFIB_TCT02",
    "SW_EXPT_EFIC_TCT02",
]
collectionB = [
    ["SW_OPER_MAGA_LR_1B", "SW_OPER_MAGA_HR_1B"],
    ["SW_OPER_MAGB_LR_1B", "SW_OPER_MAGB_HR_1B"],
    ["SW_OPER_MAGC_LR_1B", "SW_OPER_MAGC_HR_1B"]
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

        data = request.get_between(start_time, end_time, **kwargs)
        df = data.as_dataframe()
    except:
        df = []
    return df


def graphingB(index, arrayx, arrayy):
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[0].plot(arrayx, arrayy,
                 color=color[index], label="B_E, "+labels[index])
    axes[0].legend()
    axes[0].set_ylabel(r"$\Delta$  $B_{E}$ (nT)")


def graphingF(index, arrayx, arrayy):
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[2].plot(arrayx, arrayy,
                 color=color[index], label="FAC, "+labels[index])
    axes[2].legend()
    axes[2].set_ylabel("FAI Intensity (kR)")


def graphingE(index, dataset, arrayx, arrayy):
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[1].plot(arrayx, arrayy,
                 color=color[index], label=dataset+","+labels[index])
    axes[1].set_ylabel(r"$E_{N}$ (mV/m)$")
    axes[1].legend()


def requesterarraylogic():
    def B():
        def High_resolution_interpolation(time_model, b_model, proper_time):
            # creates an empty array of proper len, which has less resolution
            Bmodel_attime = np.zeros(len(proper_time))
            for i in range(len(proper_time)):
                Bmodel_attime[i] = (b_model[np.argmin(  # Optimization, finds the minimum value between the subtraction of the times, then finds the corresponding model B field
                    np.abs(proper_time[i]-time_model))])
            return Bmodel_attime

        for i in range(len(collectionB)):
            bmodel = []
            b = []
            time = []
            for k in range(2):
                ds = requester(
                    collectionB[i][k], measurements[0], asynchronous=False, show_progress=False)  # Data package level, data/modes, **kwargs
                if(k == 0):
                    Ball = ds["B_NEC"]  # data package
                    # Finds the time which is stored as a row header (ie row name)
                    time = Ball.index
                    #time = np.delete(time, -1)
                    Ball = pd.Series.to_numpy(Ball)  # turns to numpy array
                    # for i in range(len(Ball))
                    for j in range(len(Ball[:])):
                        # flattens array for the Northward data ie 2nd index
                        b.append(Ball[:][j][1])
                else:
                    pass
                if(k == 1):
                    Bmodel = ds["B_NEC_CHAOS"]
                    # print(Bmodel)
                    time_model = Bmodel.index
                    Bmodel = pd.Series.to_numpy(Bmodel)  # turns to numpy array
                    # for i in range(len(Ball))
                    for j in range(len(Bmodel[:])):
                        # flattens array for the Northward data ie 2nd index
                        bmodel.append(Bmodel[:][j][1])
                    Model = High_resolution_interpolation(
                        time_model, bmodel, time)

                else:
                    pass
            graphingB(i, time, b-Model)

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

for i in range(len(axes)):  # final touches
    mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
    # mplcyberpunk.add_gradient_fill(
    # ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")
plt.show()
