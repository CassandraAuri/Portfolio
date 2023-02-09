from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from tqdm.notebook import tqdm
set_token("https://vires.services/ows", set_default=True,
          token="kmxv5mTTyYwzw4kQ9lsCkGfQHtjjJRVZ")  # key
fig, axes = plt.subplots(nrows=3,
                         figsize=(10, 6),
                         )
plt.style.use('dark_background')
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
        request.set_products(measurements=measurement)
        data = request.get_between(start_time, end_time, **kwargs)
        df = data.as_dataframe()
    except:
        df = []
    return df


def graphingB(index, arrayx, arrayy):
    def Delta():
        delta_yarray = np.zeros(len(arrayy))
        baseline = np.linspace(arrayy[0], arrayy[-1], len(arrayy))
        for i in range(len(arrayy)):
            delta_yarray[i] = arrayy[i]-baseline[i]
        return delta_yarray
    delta_arrayy = Delta()  # assumes a linear slope, gets rid of baseline
    color = ['dodgerblue', 'purple', 'deeppink', "orange"]
    axes[0].plot(arrayx, delta_arrayy,
                 color=color[index], label="B_E, "+labels[index])
    axes[0].legend()


def graphingF(index, arrayx, arrayy):
    color = ['dodgerblue', 'purple', 'deeppink', "orange"]
    axes[2].plot(arrayx, arrayy,
                 color=color[index], label="FAC, "+labels[index])
    axes[2].legend()


def graphingE(index, dataset, arrayx, arrayy):
    color = ['dodgerblue', 'purple', 'deeppink', "orange"]
    if(dataset == "Ehx"):
        axes[1].plot(arrayx, arrayy,
                     color=color[index+2], label=dataset+","+labels[index])
    else:
        axes[1].plot(arrayx, arrayy,
                     color=color[index], label=dataset+","+labels[index])
    axes[1].legend()


def requesterarraylogic():
    def B():
        for i in range(len(collectionB)):
            b = []
            ds = requester(
                collectionB[i], measurements[0], asynchronous=False, show_progress=False)
            Ball = ds["B_NEC"]
            time = Ball.index
            Ball = pd.Series.to_numpy(Ball)

            # for i in range(len(Ball))
            for j in range(len(Ball[:])):
                b.append(Ball[:][j][1])

            graphingB(i, time, b)

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

plt.style.use('dark_background')
plt.show()
