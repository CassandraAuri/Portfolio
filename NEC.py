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
import requests
import time
import cdflib


def EBplotsNEC(dict):
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
    def moving_average(a):
        n = int(len(a)/20)  # gives 60 second averaging
        print(n, "n")
        np.savetxt('aaaa.csv', a, delimiter=',')
        y_padded = np.pad(a, (n//2, n-1-n//2), mode='edge')
        np.savetxt('padded.csv', y_padded, delimiter=',')
        y_smooth = np.convolve(y_padded, np.ones((n,))/n, mode='valid')
        np.savetxt('test.csv', y_smooth, delimiter=',')
        return y_smooth

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
        try:
            if(['Centre'] in dict["graph_PF_chosen"] == True or dict["FAC"] == True or dict["Difference"] == True):
                print("yup_23")
                length += 1
                pass
        except TypeError:
            pass
        else:
            print("yup_34")
        return length

    fig, axes = plt.subplots(nrows=rows(),
                             figsize=(15, 10), sharex=True, sharey=False
                             )

    time_range = dict["time_range"]
    has_E = []  # Sets Which space-craft have a corersponding E field
    # Labels of space-craft interested in
    labels = dict["satellite_graph"]
    # Measurement names from swarm
    measurements = ["B_NEC", [['VsatN', 'VsatE', 'VsatC'],
                              ['Evx', 'Evy', 'Evz'],  "Calibration_flags", "Quality_flags"]]
    measurements_flat = ['VsatN', 'VsatE', 'VsatC',
                         'Evx', 'Evy', 'Evz',  "Calibration_flags", "Quality_flags"]

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
    try:
        if(dict["B_frequency"][0] == "1Hz"):
            collectionB = collectionB_01
        else:
            collectionB = collectionB_50
    except TypeError:  # B is none
        collectionB = collectionB_50
    try:
        if(dict["E_frequency"][0] == "2Hz"):
            collectionE = collectionE_02
        else:
            collectionE = collectionE_16
    except TypeError:  # E is none
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
            if(dict["graph_B_chosen"][i] == "North"):
                index = 0
            elif(dict["graph_B_chosen"][i] == "East"):
                index = 1
            elif(dict["graph_B_chosen"][i] == "Centre"):
                index = 2

            print(label, index, dict["graph_B_chosen"][i])
            if(label == "e-pop"):
                nanindex = np.argwhere(np.isnan(arrayy[:, index]))
                array_y = np.delete(arrayy[:, index], nanindex)
                arrayy_average = moving_average(array_y)
                print(arrayy[:, index])
                print("divide")
                print(arrayy_average)
                axes[i+length_for_axis].plot(np.delete(arrayx, nanindex), array_y-arrayy_average,
                                             label=label)
            else:
                axes[i+length_for_axis].plot(arrayx, arrayy[:, index],
                                             label=label)
                # axes[i+length_for_axis].plot(arrayx, arrayy_average,
                #                             label=label)
                # axes[i+length_for_axis].plot(arrayx, arrayy[:, index],
                # label=label)
            axes[i+length_for_axis].legend(loc=2)
            axes[i +
                 length_for_axis].set_ylabel(r"$B_{{{}}}$".format(dict["graph_B_chosen"][i]) + " (nT) ")

    def graphingF(label, arrayx, arrayy):
        # tests if there is difference for ponyting flux and Fac, which goes in the last subplot
        try:
            if(['Centre'] in dict["graph_PF_chosen"] == True or dict["FAC"] == True or dict["Difference"] == True):
                extra_axis = 1
        except TypeError:
            extra_axis = 0
        length_for_axis = rows()-1-extra_axis

        axes[length_for_axis].plot(arrayx, arrayy, label=label
                                   )
        axes[length_for_axis].legend(loc=2)
        axes[length_for_axis].set_ylabel("FAI Intensity (kR)")

    def graphingE(label, arrayx, arrayy):
        for i in range(len(dict["graph_E_chosen"])):
            if(dict["graph_E_chosen"][i] == "North"):
                index = 0
            elif(dict["graph_E_chosen"][i] == "East"):
                index = 1
            elif(dict["graph_E_chosen"][i] == "Centre"):
                index = 2
            axes[i].plot(arrayx, arrayy[:, index],
                         label=label+'fkipped')
            axes[i].set_ylabel(
                r"$E_{{{}}}$ $(mV/m)$".format(dict["graph_E_chosen"][i]))
            axes[i].legend(loc=2)
            print(i, "reee")

    def graphingFlux(label, arrayx, arrayy):
        color = ['gold', 'cyan', 'deeppink', "red"]
        # because python starts index at 0
        length_for_axis = rows()-1-(len(dict["graph_PF_chosen"]))
        try:
            if(['Centre'] in dict["graph_PF_chosen"] == True or dict["FAC"] == True or dict["Difference"] == True):
                length_for_axis += (-1)
        except TypeError:
            pass
        print(length_for_axis)
        for i in range(len(dict["graph_PF_chosen"])):
            print(i, "wowee", dict["graph_PF_chosen"],
                  dict["graph_PF_chosen"][i], length_for_axis)
            try:
                if(dict["graph_PF_chosen"][i] == "North"):
                    index = 0
                elif(dict["graph_PF_chosen"][i] == "East"):
                    index = 1
                elif(dict["graph_PF_chosen"][i] == "Centre"):
                    index = 2
                print(index, "holy shit balls why no plots")
                axes[i+length_for_axis].plot(arrayx, arrayy[index],
                                             label=label)
                axes[i+length_for_axis].set_ylabel(
                    r"$S_{{{}}}$".format(dict["graph_PF_chosen"][i]))
                axes[i+length_for_axis].legend(loc=2)

            except IndexError:
                pass

    # [spacecraft,times,[flux,fac]]
    def graphingDifference(label, arrayx, arrayy, i):

        length_for_axis = rows()-1
        axes[length_for_axis].plot(
            arrayx, arrayy[0], label=label)
        axes[length_for_axis].set_ylabel(
            r"$S_{{{}}}$ (solid)".format("centre"))
        axes[length_for_axis].set_ylim(-1, 1)
        global ax2
        if(i == 0):
            ax2 = axes[length_for_axis].twinx()
            ax2.set_ylabel("FAC (DOTTED)")
            ax2.set_ylim(-1, 1)
        ax2.plot(
            arrayx, arrayy[1],  linestyle='dotted')
        axes[length_for_axis].legend(loc=2)

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
            satellites_with_E = []
            for i in range(len(collectionE)):
                print(has_E, "has_E")
                dsE = requester(  # requests data
                    collectionE[i], measurements_flat, False, asynchronous=False, show_progress=False)
                print(dsE['Quality_flags'].to_csv('quality.csv'))
                # QUALITY MUST BE CHECKED!!
                if(len(dsE) != 0):  # checks if empty
                    print(labels, "".join(
                        ("swarm", dsE["Spacecraft"][0].lower())), has_E)
                    # Checks if space-craft is selected
                    if("".join(("swarm", dsE["Spacecraft"][0].lower())) in labels):
                        has_E.append(True)
                        satellites_with_E.append(
                            "".join(("swarm", dsE["Spacecraft"][0].lower())))
                        print(has_E)
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
                        if(dict["graph_E_chosen"] == None):
                            pass
                        else:
                            graphingE(
                                "".join(("Swarm ", dsE["Spacecraft"][0])), dsE.index.to_numpy(), ElectricNEC)
                        # For ponyting flux
                        return_data.append(ElectricNEC)
                        returned_times = dsE.index.to_numpy()  # Times for plot

                        def B_Logic_For_E():

                            def arrangement():  # arranges B into a useable format for use later
                                barranged = np.zeros((len(Btime), 3))
                                # Re-arranges into proper (n x 3 ) matricies, ugly but works
                                for j in range(len(Btime)):
                                    for k in range(3):
                                        barranged[j][k] = Bdata[j][k]
                                return barranged

                            # finds the closest index in B for each time
                            def Time_corrections(E_time, B_time):

                                # Changes B to times found in Time_corrections
                                def Bchange(times):
                                    nonlocal Bdata
                                    Bdata = Bdata[times]
                                # finds the closest time value for each element in E_time
                                closest_time = np.zeros(
                                    len(E_time), dtype=int)
                                # resource intesiive, find something better
                                print("profile 1")
                                print(E_time[0], B_time)
                                for i in range(len(E_time)):
                                    closest_time[i] = np.argmin(
                                        np.abs(E_time[i]-B_time))

                                Bchange(closest_time)
                                print("profile2")
                                return closest_time
                            print("b_test")
                            dsB = requester(
                                collectionB[i], measurements[0], False, asynchronous=False, show_progress=False)
                            Bdata = dsB["B_NEC"].to_numpy()
                            Btime = dsB.index
                            print(dsB, "help me")
                            Bdata = arrangement()
                            times_of_b_for_flux = Time_corrections(
                                dsE.index.to_numpy(), dsB.index.to_numpy())  # Takes times of both E and B and finds the closest values in B to E

                            return times_of_b_for_flux
                        print("profile")
                        times_for_flux = B_Logic_For_E()
                        print("wtf")
                    else:
                        # Says theres no E component
                        has_E.append(False)
                else:  # Says theres no E component
                    has_E.append(False)

            return return_data, times_for_flux, returned_times, satellites_with_E

        def B():
            print("are you working", has_E)
            return_data = []
            # Goes through every satellite
            for i in range(len(dict["satellite_graph"])):
                if(dict["satellite_graph"][i] == "epop"):

                    cdf_file = cdflib.CDF(
                        r"C:\Users\1101w\Clone\Programming-8\Summer 2023\SW_OPER_MAGE_HR_1B_20210318T000000_20210318T235959_0201_MDR_MAG_HR.cdf")
                    time = cdf_file.varget('Timestamp')
                    utc = cdflib.cdfepoch.to_datetime(time)
                    utc = np.array(utc)

                    #x_start, x_end = utc-time_range[0], utc-time_range[1]

                    operation = np.argmin(
                        (np.vectorize(lambda x: x.total_seconds())))
                    print(utc-time_range[0])
                    time_closest_start, time_closest_end = np.argmin(np.abs(pd.Series(
                        utc-time_range[0]).dt.total_seconds().to_numpy())), np.argmin(np.abs(pd.Series(
                            utc-time_range[1]).dt.total_seconds().to_numpy()))
                    print(time_closest_start, time_closest_end)
                    selected_times = np.linspace(
                        time_closest_start, time_closest_end, time_closest_end-time_closest_start+1, dtype=int)
                    print(selected_times)
                    #latitude = cdf_file.varget('Latitude')
                    #longitude = cdf_file.varget('Longitude')
                    #radius = cdf_file.varget('Radius')
                    b = cdf_file.varget('B_NEC_out')
                    b_model = cdf_file.varget('B_model_NEC')
                    q_flags = cdf_file.varget('Flags_q')
                    b_flags = cdf_file.varget('Flags_B')

                    # total_residual = np.sqrt((b[selected_times, 0] - b_model[selected_times, 0])**2 + (
                    # b[selected_times, 1] - b_model[selected_times, 1])**2 + (b[selected_times, 2] - b_model[selected_times, 2])**2)
                    b = b[selected_times]
                    b_model = b_model[selected_times]
                    b[:, 0] = b[:, 0] - b_model[:, 0]
                    b[:, 1] = b[:, 1] - b_model[:, 1]
                    b[:, 2] = b[:, 2] - b_model[:, 2]

                    graphingB("e-pop", utc[selected_times],
                              b)
                else:
                    # Data package level, data/modes, **kwargs
                    if(dict["satellite_graph"][i] == "swarma"):
                        j = 0
                    if(dict["satellite_graph"][i] == "swarmb"):
                        j = 1
                    if(dict["satellite_graph"][i] == "swarmc"):
                        j = 2
                    dsmodel_res = requester(collectionB[j], measurements[0], True,
                                            asynchronous=False, show_progress=False)
                    if("".join(("swarm", dsmodel_res["Spacecraft"][0].lower())) in labels):
                        ds = requester(collectionB[j], measurements[0], False,
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
                        bresarranged = np.zeros((len(b), 3))
                        # Re-arranges into proper (n x 3 ) matricies, ugly but works
                        for l in range(len(b)):
                            for k in range(3):
                                barranged[l][k] = b[l][k]
                                bmodelarranged[l][k] = bmodel_actual[l][k]
                        bresarranged = np.subtract(barranged, bmodelarranged)
                        if(dict["graph_B_chosen"] == None):
                            pass
                        else:
                            graphingB(
                                "".join(("Swarm ", dsmodel_res["Spacecraft"][0])), time, bresarranged)

                        # graphingB(i, time, datamfa[:,0]) #Collects all the compressional B values
                        # Returns magnetic field for poynting flux if there is a corresponding E field
                        print(has_E, i, "flawop")
                        # only needs to pass data back if we need to look at pyonting flux
                        if(dict["graph_PF_chosen"] == None):
                            pass
                        else:
                            if(has_E[j] == True):
                                return_data.append(bresarranged)
                    print(return_data)
            else:
                pass
            return return_data

        def F(space_craft_with_E):
            data_return = []
            for i in range(len(collectionF)):
                ds = requester(
                    collectionF[i], "FAC", False, asynchronous=False, show_progress=False)
                if("".join(("swarm", ds["Spacecraft"][0].lower())) in labels):
                    fac = ds["FAC"]
                    time = fac.index
                    fac = pd.Series.to_numpy(fac)

                    graphingF(
                        "".join(("Swarm ", ds["Spacecraft"][0])), time, fac)
                    if("".join(("swarm", ds["Spacecraft"][0].lower())) in space_craft_with_E):
                        data_return.append(
                            ["".join(("Swarm ", ds["Spacecraft"][0])), time.to_numpy(), fac])
                else:
                    pass
            return data_return
        if(dict["graph_E_chosen"] == None and dict["graph_PF_chosen"] == None):
            pass
        else:
            # E field, indexs of B for time, times for plotting flux
            efield, times_for_b, time_for_flux, space_craft_with_E = E()
        if(dict["graph_B_chosen"] == None and dict["graph_PF_chosen"] == None):
            pass
        else:

            bfield = B()

        if(dict["FAC"] == True):
            FAC_data = F(space_craft_with_E)

        def pontying_flux():  # Take rough estimate by doing N cross E to get C poynting flux
            nonlocal bfield, efield
            return_data = []

            for i in range(len(efield)):  # Sees how many E data points we have
                bflux = bfield[i]
                bflux = bflux[times_for_b]

                eflux = efield[i]

                flux = np.cross(eflux, bflux[i])
                flux_individual = np.transpose(flux)
                graphingFlux(
                    space_craft_with_E[i], time_for_flux, flux_individual)
                return_data.append(
                    [space_craft_with_E[i], time_for_flux, flux_individual[2]])
            return return_data

        # [ spacecraft_name, times_flux, data], [space_craft_name, time_FAC,data]
        def Difference_plots(flux, fac):
            # checks length of FAC versus flux, if flux is 16Hz, change to 2Hz to match FAC
            if(len(flux[0][1]) == len(fac[0][1])):
                pass
            else:
                closest_time = np.zeros(len(fac[0][1]), dtype=int)
                # resource intesiive, find something better

                for i in range(len(fac[0][1])):
                    closest_time[i] = np.argmin(
                        np.abs(fac[0][1][i]-flux[0][1]))  # finds the closest value of time_flux for each fac_time
                for i in range(len(flux)):
                    flux_time_corrected = flux[i][2][closest_time]
                    graphingDifference(flux[i][0], fac[i][1], [
                        flux_time_corrected/np.max(flux_time_corrected), np.array(fac[i][2])/np.max(fac[i][2])], i)

        if(dict["graph_PF_chosen"] == None):
            pass
        else:
            flux = pontying_flux()
        try:
            if(['Centre'] in dict["graph_PF_chosen"] == True or dict["FAC"] == True or dict["Difference"] == True):
                Difference_plots(flux, FAC_data)
        except TypeError:
            pass

    requesterarraylogic()

    fig.supxlabel("Time (Day:Hour:Minute)")
    fig.suptitle("Time Versus Auroral Parameters From Swarm Spacecraft")

    # for i in range(len(axes)):  # final touches
    # mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
    # mplcyberpunk.add_gradient_fill(
    # ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")

    return fig, axes
