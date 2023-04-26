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

time_range = (datetime(2021, 3, 18, 8, 12),
              datetime(2021, 3, 18, 8, 22))

global has_E
has_E = []  # Sets Which space-craft have a corersponding E field
# Labels of space-craft interested in
labels = ["Swarm A", "Swarm B", "Swarm C"]
# Measurement names from swarm
measurements = ["B_NEC", ['VsatN', 'VsatE', 'VsatC']]


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
    "SW_OPER_FACCTMS_2F"]  # Data packages from swarm


def requester(sc_collection, measurement, residual, **kwargs):  # Requests data from swarm
    try:
        request = SwarmRequest()
        request.set_collection(sc_collection)
        if(residual == True):
            request.set_products(measurements=measurement, models=[
                "CHAOS"], residuals=True)
        else:
            request.set_products(measurements=measurement, models=[
                "CHAOS"])
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


def graphingE(index, arrayx, arrayy):
    color = ['gold', 'cyan', 'deeppink', "red"]
    axes[1].plot(arrayx, arrayy,
                 color=color[index], label=labels[index])
    axes[1].set_ylabel(r"$E_{N}$ $(mV/m)$")
    axes[1].legend(loc=2)


def Coordinate_change(lattiude, longitude, radius):  # Coordinate change
    a, b, e2 = 6378137.0, 6356752.3142, 0.00669437999014  # From DRS80
    lat, lon, h = lattiude, longitude, radius
    v = a/np.sqrt(1-e2*np.sin(lat)*np.sin(lat))  # logic
    x = (v+h)*np.cos(lat)*np.cos(lon)
    y = (v+h)*np.cos(lat)*np.sin(lon)
    z = (v*(1-e2)+h)*np.sin(lat)
    return [x, y, z]


def requesterarraylogic():
    def E():
        return_data = []

        for i in range(len(collectionE)):

            dsE = requester(
                collectionE[i], measurements[1], False, asynchronous=False, show_progress=False)
            # B is 50hz but E is 16 so we must do corrections grrr

            if(len(dsE) != 0):

                print(i)
                # Tells B how many E's there are for ponyting flux,
                has_E.append(True)
                print(has_E, "# of e")
                dsB = requester(
                    collectionB[i], measurements[0], False, asynchronous=False, show_progress=False)
                Bdata = dsB["B_NEC"].to_numpy()
                Btime = dsB.index
                Bmodel = dsB["B_NEC_CHAOS"]

                def arrangement():
                    barranged = np.zeros((len(Btime), 3))
                    bmodelarranged = np.zeros((len(Btime), 3))
                    # Re-arranges into proper (n x 3 ) matricies, ugly but works
                    for j in range(len(Btime)):
                        for k in range(3):
                            barranged[j][k] = Bdata[j][k]
                            bmodelarranged[j][k] = Bmodel[j][k]
                    return barranged, bmodelarranged

                Bdata, Bmodel = arrangement()
                Velocitytime = dsE.index.to_numpy()

                def Time_corrections(E_time, B_time):
                    def Bchange(times):
                        print(times)
                        nonlocal Bdata, Bmodel
                        Bdata = Bdata[times]
                        Bmodel = Bmodel[times]

                    # finds the closest time value for each element in E_time
                    closest_time = np.zeros(len(E_time), dtype=int)
                    # resource intesiive, find something better

                    for i in range(len(E_time)):
                        closest_time[i] = np.argmin(
                            np.abs(E_time[i]-B_time))

                    Bchange(closest_time)
                    return closest_time
                times_of_b_for_flux = Time_corrections(
                    Velocitytime, dsB.index.to_numpy())
                Velocity = dsE[measurements[1]].to_numpy()
                # Via E=-(v cross B)
                bdata_units, Bmodel_units = np.multiply(
                    Bdata, 10**(-9)), np.multiply(Bmodel, 10**(-9))
                Edata = np.divide(
                    np.cross(Velocity, bdata_units), -1*1.602*10**(-19))  # Gives E in NEC via lorentz force
                Emodel = np.divide(
                    np.cross(Velocity, Bmodel_units), -1*1.602*10**(-19))  # Gives E in NEC
                Bres = np.subtract(Bdata, Bmodel)

                radius, lattiude, longitude = dsE["Radius"].to_numpy(
                ), dsE['Latitude'].to_numpy(), dsE['Longitude'].to_numpy()  # Gets Emphermis data
                r_nec = Coordinate_change(lattiude, longitude, radius)
                datamfa = MFA(Edata, Bres, np.asarray(
                    r_nec).T)  # Calls MFA with (3xn) vectors
                modelmfa = MFA(Emodel, Bres, np.asarray(
                    r_nec).T)

                graphingE(i, dsE.index.to_numpy(), np.subtract(
                    datamfa[:, 1], modelmfa[:, 1]))
                return_data.append(np.subtract(
                    datamfa, modelmfa))
                returned_times = dsE.index.to_numpy()
            else:  # Says theres no E component
                has_E.append(False)
        print(np.shape(return_data))
        return return_data, times_of_b_for_flux, returned_times

    def B():
        return_data = []
        for i in range(len(collectionB)):  # Goes through every satellite
            # Data package level, data/modes, **kwargs
            dsmodel_res = requester(collectionB[i], measurements[0], True,
                                    asynchronous=False, show_progress=False)
            ds = requester(collectionB[i], measurements[0], False,
                           asynchronous=False, show_progress=False)

            def model():  # Gets B field from CHAOS model for Mean-Field
                Bmodel_res = dsmodel_res["B_NEC_res_CHAOS"]
                Bmodel = ds["B_NEC_CHAOS"]
                return Bmodel_res.to_numpy(), Bmodel.to_numpy()
            bmodel_res, bmodel_actual = model()
            radius, lattiude, longitude = ds["Radius"].to_numpy(
            ), ds['Latitude'].to_numpy(), ds['Longitude'].to_numpy()  # Gets Emphermis data
            Bdata = ds["B_NEC"]  # data package
            # Finds the time which is stored as a row header (ie row name)
            time = Bdata.index
            # time = np.delete(time, -1)
            b = Bdata.to_numpy()
            # since minutes only, could start half way nbetween a measurement
            r_nec = Coordinate_change(lattiude, longitude, radius)
            barranged = np.zeros((len(b), 3))
            bmodelarranged = np.zeros((len(b), 3))
            bmodelresarranged = np.zeros((len(b), 3))
            for j in range(len(b)):  # Re-arranges into proper (n x 3 ) matricies, ugly but works
                for k in range(3):
                    barranged[j][k] = b[j][k]
                    bmodelresarranged[j][k] = bmodel_res[j][k]
                    bmodelarranged[j][k] = bmodel_actual[j][k]
            datamfa = MFA(barranged, bmodelresarranged, np.asarray(
                r_nec).T)  # Calls MFA with (3xn) vectors
            bmodelmfa = MFA(bmodelarranged, bmodelresarranged, np.asarray(
                r_nec).T)
            graphingB(i, time, np.subtract(datamfa[:, 2], bmodelmfa[:, 2]))
            # graphingB(i, time, datamfa[:,2])
            # graphingB(i, time, datamfa[:,0]) #Collects all the compressional B values
            # Returns magnetic field for poynting flux if there is a corresponding E field
            if(has_E[i] == True):
                return_data.append(np.subtract(datamfa, bmodelmfa))
                print(return_data)
                print("bfieldarrays")

        return np.array(return_data)

    def F():
        for i in range(len(collectionF)):
            b = []
            ds = requester(
                collectionF[i], "FAC", False, asynchronous=False, show_progress=False)
            fac = ds["FAC"]
            time = fac.index
            fac = pd.Series.to_numpy(fac)

            graphingF(i, time, fac)

    e1, e2, e3 = E()  # E field, indexs of B for time, times for plotting flux
    efield, times_for_b, time_for_flux = e1, e2, e3

    bfield = B()

    F()
    plt.show()
    plt.clf()

    def pontying_flux():  # Take rough estimate by doing N cross E to get C poynting flux
        nonlocal bfield, efield
        flux = []
        print(np.shape(bfield[0]))
        for i in range(len(efield)):  # Sees how many E data points we have
            bflux = bfield[i]
            print(bflux)
            bflux = bflux[times_for_b]
            print(bflux)
            print(np.shape(bflux))
            eflux = efield[i]
            print(np.shape(eflux), "test")
            flux = np.cross(eflux, bflux[i])
            print(np.shape(flux))

            for i in range(3):
                plt.clf()
                flux_individual = np.transpose(flux)[i]
                plt.plot(time_for_flux, flux_individual)
                plt.show()
                plt.clf()

    pontying_flux()
    print("test")


requesterarraylogic()

fig.supxlabel("Time (Day:Hour:Minute)")
fig.suptitle("Time Versus Auroral Parameters From Swarm Spacecraft")

# for i in range(len(axes)):  # final touches
# mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
# mplcyberpunk.add_gradient_fill(
# ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")
plt.show()
