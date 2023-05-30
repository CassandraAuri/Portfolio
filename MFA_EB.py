from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
import mplcyberpunk
from itertools import chain
from MFA import MFA
import pickle


def EBplotsMFA(dict):
    set_token("https://vires.services/ows", set_default=True,
              token="kmxv5mTTyYwzw4kQ9lsCkGfQHtjjJRVZ")  # key
    plt.style.use("cyberpunk")  # Dark mode!
    global has_E
    print(dict)
    """
    if(dict["graph_B_chosen"] == ['']):
        dict["graph_B_chosen"] = []
    else:
        pass

    if(dict["graph_E_chosen"] == ['']):
        dict["graph_E_chosen"] = []
        print('yay')
    else:
        pass

    if(dict['graph_PF_chosen'] == ['']):
        dict['graph_PF_chosen'] = []
    else:
        pass
    """

    def rows():  # finds the number of rows needed for the produced figure
        length = 0
        try:
            length += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        try:
            length += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length += len(dict["graph_PF_chosen"])
        except TypeError:
            pass
        if(dict["FAC"] == True):
            length += 1
        print(length)
        return length

    fig, axes = plt.subplots(nrows=rows(),
                             figsize=(10, 7), sharex=True, sharey=False
                             )

    time_range = dict["time_range"]
    has_E = []  # Sets Which space-craft have a corersponding E field
    # Labels of space-craft interested in
    labels = dict["satellite_graph"]
    # Measurement names from swarm
    measurements = ["B_NEC", [['VsatN', 'VsatE', 'VsatC'],
                              ['Ehx', 'Ehy', 'Ehz'],  'Quality_flags']]
    measurements_flat = ['VsatN', 'VsatE', 'VsatC',
                         'Ehx', 'Ehy', 'Ehz',  'Quality_flags']

    collectionE_16 = [
        "SW_EXPT_EFIA_TCT16",
        "SW_EXPT_EFIB_TCT16",
        "SW_EXPT_EFIC_TCT16",
    ]
    collectionE_02 = [
        "SW_EXPT_EFIA_TCT02",
        "SW_EXPT_EFIB_TCT02",
        "SW_EXPT_EFIC_TCT02",
    ]
    collectionB_50 = [
        "SW_OPER_MAGA_HR_1B",
        "SW_OPER_MAGB_HR_1B",
        "SW_OPER_MAGC_HR_1B"
    ]
    collectionB_01 = [
        "SW_OPER_MAGA_LR_1B",
        "SW_OPER_MAGB_LR_1B",
        "SW_OPER_MAGC_LR_1B"
    ]
    collectionF = [
        "SW_OPER_FACATMS_2F",
        "SW_OPER_FACBTMS_2F",
        "SW_OPER_FACCTMS_2F"]  # Data packages from swarm

    # IF B, not selected but ponyting flux is, we assume 50hz data
    if(dict["B_frequency"][0] == "1Hz"):
        collectionB = collectionB_01
    else:
        collectionB = collectionB_50
    print(dict["E_frequency"][0], "woweeeeee")
    if(dict["E_frequency"][0] == "2Hz"):
        collectionE = collectionE_02
    else:
        collectionE = collectionE_16

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

    def graphingB(label, arrayx, arrayy):
        color = ['gold', 'cyan', 'deeppink', "red"]
        length_for_axis = len(dict["graph_E_chosen"])
        for i in range(len(dict["graph_B_chosen"])):
            try:
                if(dict["graph_B_chosen"][i] == "Polodial"):
                    index = 0
                elif(dict["graph_B_chosen"][i] == "Azimuthal"):
                    index = 1
                elif(dict["graph_B_chosen"][i] == "Mean-field"):
                    index = 2

                axes[i+length_for_axis].plot(arrayx, arrayy[:, index],
                                             label=label)
                axes[i+length_for_axis].legend(loc=2)
                axes[i +
                     length_for_axis].set_ylabel(r"$B_{{{}}}$".format(dict["graph_B_chosen"][i]) + " (nT) ")
                print(i+length_for_axis)
            except IndexError:
                pass

    def graphingF(label, arrayx, arrayy):
        length_for_axis = rows()-1
        print(length_for_axis, "pppoggo")

        axes[length_for_axis].plot(arrayx, arrayy, label=label
                                   )
        axes[length_for_axis].legend(loc=2)
        axes[length_for_axis].set_ylabel("FAI Intensity (kR)")

    def graphingE(label, arrayx, arrayy):
        for i in range(len(dict["graph_E_chosen"])):
            try:
                if(dict["graph_E_chosen"][i] == "Polodial"):
                    index = 0
                elif(dict["graph_E_chosen"][i] == "Azimuthal"):
                    index = 1
                elif(dict["graph_E_chosen"][i] == "Mean-field"):
                    index = 2
                axes[i].plot(arrayx, arrayy[:, index],
                             label=label)
                axes[i].set_ylabel(
                    r"$E_{{{}}}$ $(mV/m)$".format(dict["graph_E_chosen"][i]))
                axes[i].legend(loc=2)
                print(i, "reee")
            except IndexError:
                pass

    def graphingFlux(label, arrayx, arrayy):
        color = ['gold', 'cyan', 'deeppink', "red"]
        # because python starts index at 0
        length_for_axis = rows()-1-(len(dict["graph_PF_chosen"]))
        print(length_for_axis)
        for i in range(len(dict["graph_PF_chosen"])):
            try:
                if(dict["graph_PF_chosen"][i] == "Polodial"):
                    index = 0
                elif(dict["graph_PF_chosen"][i] == "Azimuthal"):
                    index = 1
                elif(dict["graph_PF_chosen"][i] == "Mean-field"):
                    index = 2
                print(index, "holy shit balls why no plots")
                axes[i+length_for_axis].plot(arrayx, arrayy[index],
                                             label=labels[label])
                axes[i+length_for_axis].set_ylabel(
                    r"$S_{{{}}}$".format(dict["graph_PF_chosen"][label]))
                axes[i+length_for_axis].legend(loc=2)
                print(i+length_for_axis)
            except IndexError:
                pass

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
            returned_times = []
            for i in range(len(collectionE)):
                dsE = requester(  # requests data
                    collectionE[i], measurements_flat, False, asynchronous=False, show_progress=False)

                # QUALITY MUST BE CHECKED!!

                if(len(dsE) != 0):  # checks if empty
                    print(labels, "".join(("Swarm ", dsE["Spacecraft"][0])))
                    # Checks if space-craft is selected
                    if("".join(("Swarm ", dsE["Spacecraft"][0])) in labels):
                        has_E.append(True)
                        print("holy shit bucko, its pogged out",
                              "".join(("Swarm ", dsE["Spacecraft"][0])))
                        Velocity = dsE[measurements[1][0]
                                       ].to_numpy()  # Gets veloctes for lorentz force
                        VelocitySum = Velocity.sum(axis=1)
                        # Normalizes and finds unitary
                        Velocity_Unit = Velocity / \
                            VelocitySum[:, np.newaxis]  # normalizes

                        # Gets electric field in XYZ coordiinates
                        # Gets electric field data
                        Electric = dsE[measurements[1][1]].to_numpy()
                        ElectricNEC = np.multiply(
                            Velocity_Unit, Electric)  # transforms into NEC

                        # Plots electric field time seres
                        print(dict["graph_E_chosen"])

                        return_data.append(ElectricNEC)  # For ponyting flux
                        returned_times = dsE.index.to_numpy()  # Times for plot

                        def B_Logic_For_E():

                            def arrangement():  # arranges B into a useable format for use later
                                barranged = np.zeros((len(Btime), 3))
                                bmodelarranged = np.zeros((len(Btime), 3))
                                # Re-arranges into proper (n x 3 ) matricies, ugly but works
                                for j in range(len(Btime)):
                                    for k in range(3):
                                        barranged[j][k] = Bdata[j][k]
                                        bmodelarranged[j][k] = Bdata[j][k]
                                return barranged, bmodelarranged

                            # finds the closest index in B for each time
                            def Time_corrections(E_time, B_time):

                                # Changes B to times found in Time_corrections
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

                            dsB = requester(
                                collectionB[i], measurements[0], False, asynchronous=False, show_progress=False)
                            Bdata = dsB["B_NEC"].to_numpy()
                            Btime = dsB.index
                            BModel = dsB["B_NEC_CHAOS"]
                            Bdata, Bmodel = arrangement()
                            times_of_b_for_flux = Time_corrections(
                                dsE.index.to_numpy(), dsB.index.to_numpy())  # Takes times of both E and B and finds the closest values in B to E
                            return times_of_b_for_flux, Bmodel

                        times_for_flux, MFA_vector = B_Logic_For_E()

                        radius, lattiude, longitude = dsE["Radius"].to_numpy(
                        ), dsE['Latitude'].to_numpy(), dsE['Longitude'].to_numpy()  # Gets Emphermis data
                        r_nec = Coordinate_change(lattiude, longitude, radius)
                        B
                        print(ElectricNEC, "test me daddy")
                        print(MFA_vector, "pog")
                        datamfa = MFA(ElectricNEC, MFA_vector, np.asarray(
                            r_nec).T)  # Calls MFA with (3xn) vectors

                        return_data.append(
                            datamfa)
                        returned_times = dsE.index.to_numpy()
                        if(dict["graph_E_chosen"] == None):
                            pass
                        else:
                            print("holy shit, its snek")
                            graphingE(
                                "".join(("Swarm ", dsE["Spacecraft"][0])), dsE.index.to_numpy(), datamfa)
                    else:
                      # Says theres no E component
                        has_E.append(False)
                else:  # Says theres no E component
                    has_E.append(False)
            return return_data, times_for_flux, returned_times

        def B():
            print("are you working")
            return_data = []
            for i in range(len(collectionB)):  # Goes through every satellite
                # Data package level, data/modes, **kwargs
                dsmodel_res = requester(collectionB[i], measurements[0], True,
                                        asynchronous=False, show_progress=False)
                if("".join(("Swarm ", dsmodel_res["Spacecraft"][0])) in labels):
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
                    # Re-arranges into proper (n x 3 ) matricies, ugly but works
                    for j in range(len(b)):
                        for k in range(3):
                            barranged[j][k] = b[j][k]
                            bmodelresarranged[j][k] = bmodel_res[j][k]
                            bmodelarranged[j][k] = bmodel_actual[j][k]
                    datamfa = MFA(bmodelresarranged, bmodelarranged, np.asarray(
                        r_nec).T)  # Calls MFA with (3xn) vectors
                    if(dict["graph_B_chosen"] == None):
                        pass
                    else:
                        graphingB(
                            "".join(("Swarm ", dsmodel_res["Spacecraft"][0])), time, datamfa)

                    # graphingB(i, time, datamfa[:,0]) #Collects all the compressional B values
                    # Returns magnetic field for poynting flux if there is a corresponding E field
                    print(has_E, i, "flawop")
                    # only needs to pass data back if we need to look at pyonting flux
                    if(dict["graph_PF_chosen"] == None):
                        pass
                    else:
                        if(has_E[i] == True):
                            return_data.append(datamfa)
                print(return_data)
            else:
                pass
            return return_data

        def F():
            for i in range(len(collectionF)):

                ds = requester(
                    collectionF[i], "FAC", False, asynchronous=False, show_progress=False)
                if("".join(("Swarm ", ds["Spacecraft"][0])) in labels):
                    fac = ds["FAC"]
                    time = fac.index
                    fac = pd.Series.to_numpy(fac)

                    graphingF(
                        "".join(("Swarm ", ds["Spacecraft"][0])), time, fac)
                else:
                    pass
        if(dict["graph_E_chosen"] == None and dict["graph_PF_chosen"] == None):
            pass
        else:
            e1, e2, e3 = E()  # E field, indexs of B for time, times for plotting flux
            efield, times_for_b, time_for_flux = e1, e2, e3
        print(dict["graph_B_chosen"], dict["graph_PF_chosen"])
        if(dict["graph_B_chosen"] == None and dict["graph_PF_chosen"] == None):
            pass
        else:
            print("test_test")
            bfield = B()
        print(dict["FAC"])
        if(dict["FAC"] == True):
            F()

        def pontying_flux():  # Take rough estimate by doing N cross E to get C poynting flux
            nonlocal bfield, efield
            flux = []

            for i in range(len(efield)):  # Sees how many E data points we have
                bflux = bfield[i]
                bflux = bflux[times_for_b]

                eflux = efield[i]

                flux = np.cross(eflux, bflux[i])
                flux_individual = np.transpose(flux)
                graphingFlux(i, time_for_flux, flux_individual)
        print(dict["graph_PF_chosen"])
        if(dict["graph_PF_chosen"] == None):
            pass
        else:
            pontying_flux()
            print("test fucky wucky booboo")

    requesterarraylogic()

    fig.supxlabel("Time (Day:Hour:Minute)")
    fig.suptitle("Time Versus Auroral Parameters From Swarm Spacecraft")

    # for i in range(len(axes)):  # final touches
    # mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
    # mplcyberpunk.add_gradient_fill(
    # ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")

    return fig
