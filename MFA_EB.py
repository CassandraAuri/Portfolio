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
from scipy.fft import fft, fftfreq
import asilib
import asilib.asi


def EBplotsMFA(dict):

    set_token("https://vires.services/ows", set_default=True,
              token="kmxv5mTTyYwzw4kQ9lsCkGfQHtjjJRVZ")  # key
    plt.style.use("cyberpunk")  # Dark mode!
    global has_E
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

    def empharisis_processing(cadence):
        emph = []
        space_craft = []
        data_stuff = []
        for i in range(len(collectionB_01)):
            ds = requester(
                collectionB_01[i], ["F"], False, asynchronous=False, show_progress=False)  # , sampling_step="PT{}S".format(cadence))
            if("".join(("swarm", ds["Spacecraft"][0].lower())) in labels):
                data_stuff.append(ds)
            else:
                pass
        for i in range(len(data_stuff)):

            time_array = data_stuff[i]['Spacecraft'].index
            # Since time array is one hertz and we want to look for 1/cadence hertz we simply use
            lat_satellite_not_footprint = data_stuff[i]['Latitude'].to_numpy()
            lon_satellite_not_footprint = data_stuff[i]['Longitude'].to_numpy()
            altitude = data_stuff[i]['Radius'].to_numpy()/1000-6378.1370
            delete_array = np.linspace(0, cadence-1, cadence, dtype=int)
            emph.append([np.delete(time_array, delete_array), np.delete(lat_satellite_not_footprint, delete_array), np.delete(
                lon_satellite_not_footprint, delete_array), np.delete(altitude, delete_array)])
            # emph.append([time_array, lat_satellite_not_footprint,
            # lon_satellite_not_footprint, altitude])
            space_craft.append(data_stuff[i]['Spacecraft'].to_numpy()[0])
        return emph, space_craft

    def _mask_low_horizon(lon_map, lat_map, el_map, min_elevation, image=None):

        # Mask the image, skymap['lon'], skymap['lat'] arrays with np.nans
        # where the skymap['el'] < min_elevation or is nan.

        idh = np.where(np.isnan(el_map) |
                       (el_map < min_elevation))
        # Copy variables to not modify original np.arrays.
        lon_map_copy = lon_map.copy()
        lat_map_copy = lat_map.copy()
        lon_map_copy[idh] = np.nan
        lat_map_copy[idh] = np.nan

        if image is not None:
            image_copy = image.copy()
            # Can't mask image unless it is a float array.
            image_copy = image_copy.astype(float)
            image_copy[idh] = np.nan
        else:
            image_copy = None

            if (lon_map.shape[0] == el_map.shape[0] + 1) and (lon_map.shape[1] == el_map.shape[1] + 1):
                # TODO: This is REGO/THEMIS specific. Remove here and add this to the themis() function?
                # For some reason the THEMIS & REGO lat/lon_map arrays are one size larger than el_map, so
                # here we mask the boundary indices in el_map by adding 1 to both the rows
                # and columns.
                idh_boundary_bottom = (
                    idh[0] + 1,
                    idh[1],
                )  # idh is a tuple so we have to create a new one.
                idh_boundary_right = (idh[0], idh[1] + 1)
                lon_map_copy[idh_boundary_bottom] = np.nan
                lat_map_copy[idh_boundary_bottom] = np.nan
                lon_map_copy[idh_boundary_right] = np.nan
                lat_map_copy[idh_boundary_right] = np.nan

        x, y = lon_map_copy, lat_map_copy
        mask = np.isfinite(x) & np.isfinite(y)
        top = None
        bottom = None

        for i, m in enumerate(mask):
            # A common use for nonzero is to find the indices of
            # an array, where a condition is True (not nan or inf)
            good = m.nonzero()[0]

            if good.size == 0:  # Skip row is all columns are nans.
                continue
            # First row that has at least 1 valid value.
            elif top is None:
                top = i
            # Bottom row that has at least 1 value value. All indices in between top and bottom
            else:
                bottom = i

            # Reassign all lat/lon columns after good[-1] (all nans) to good[-1].
            x[i, good[-1]:] = x[i, good[-1]]
            y[i, good[-1]:] = y[i, good[-1]]
            # Reassign all lat/lon columns before good[0] (all nans) to good[0].
            x[i, : good[0]] = x[i, good[0]]
            y[i, : good[0]] = y[i, good[0]]

            # Reassign all of the fully invalid lat/lon rows above top to the the max lat/lon value.
        x[:top, :] = np.nanmax(x[top, :])
        y[:top, :] = np.nanmax(y[top, :])
        # Same, but for the rows below bottom.
        x[bottom:, :] = np.nanmax(x[bottom, :])
        y[bottom:, :] = np.nanmax(y[bottom, :])

        # converts to magnetic coordinates
        # for i in range(len(x)):
        # for i in range(len(x)):
        # y[i], x[i], ignored = aacgmv2.convert_latlon_arr(
        # y[i], x[i], alt, time_range[0], method_code='G2A')
        return x, y, image_copy

    def moving_average(a):
        n = int(len(a)/20)  # gives 60 second averaging
        y_padded = np.pad(a, (n//2, n-1-n//2), mode='edge')
        y_smooth = np.convolve(y_padded, np.ones((n,))/n, mode='valid')
        return y_smooth

    def rows():  # finds the number of rows needed for the produced figure
        length_for_axis = 0
        try:
            length_for_axis += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_PF_chosen"])
        except TypeError:
            pass
        if(dict['FAC'] == True):
            length_for_axis += 1
        else:
            pass
        if(dict['Difference'] == True):
            length_for_axis += 1
        else:
            pass
        if(dict['E_B_ratio'] == True):
            length_for_axis += 1
        try:
            length_for_axis += len(dict['sky_map_values'])
        except TypeError:
            pass
        return length_for_axis

    fig, axes = plt.subplots(nrows=rows(),
                             figsize=(15, 10), sharex=False, sharey=False, constrained_layout=True
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

    # Requests data from swarm
    def requester(sc_collection, measurement, residual, sampling_step=None, **kwargs):
        try:
            request = SwarmRequest()
            request.set_collection(sc_collection)
            if(residual == True):
                request.set_products(measurements=measurement, models=[
                    "CHAOS"], residuals=True, sampling_step=sampling_step)
            else:
                request.set_products(measurements=measurement, models=[
                    "CHAOS"], sampling_step=sampling_step)
            data = request.get_between(time_range[0], time_range[1], **kwargs)
            df = data.as_dataframe()
        except:
            df = []
        return df

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
                axes[i].set_xlim((time_range[0], time_range[1]))
            except IndexError:
                pass

    def graphingB(label, arrayx, arrayy):
        try:
            length_for_axis = len(dict["graph_E_chosen"])
        except TypeError:
            length_for_axis = 0
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
                axes[i +
                     length_for_axis].set_xlim((time_range[0], time_range[1]))
            except IndexError:
                pass

    def graphingFlux(label, arrayx, arrayy):
        length_for_axis = 0
        # because python starts index at 0
        try:
            length_for_axis += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        for i in range(len(dict["graph_PF_chosen"])):
            try:
                if(dict["graph_PF_chosen"][i] == "Polodial"):
                    index = 0
                elif(dict["graph_PF_chosen"][i] == "Azimuthal"):
                    index = 1
                elif(dict["graph_PF_chosen"][i] == "Mean-field"):
                    index = 2

                axes[i+length_for_axis].plot(arrayx, arrayy[index],
                                             label=label)
                axes[i+length_for_axis].set_ylabel(
                    r"$S_{{{}}}$".format(dict["graph_PF_chosen"][i]))
                axes[i+length_for_axis].legend(loc=2)
                axes[i +
                     length_for_axis].set_xlim((time_range[0], time_range[1]))
            except IndexError:
                pass

    def graphingF(label, arrayx, arrayy):
        length_for_axis = 0

        try:
            length_for_axis += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_PF_chosen"])
        except TypeError:
            pass

        axes[length_for_axis].plot(arrayx, arrayy, label=label
                                   )
        axes[length_for_axis].legend(loc=2)
        axes[length_for_axis].set_ylabel("FAI Intensity (kR)")
        axes[length_for_axis].set_xlim((time_range[0], time_range[1]))

    def graphingDifference(label, arrayx, arrayy, i):
        length_for_axis = 0
        try:
            length_for_axis += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_PF_chosen"])
        except TypeError:
            pass
        if(dict['FAC'] == True):
            length_for_axis += 1
        else:
            pass

        if(label == "swarma"):
            axes[length_for_axis].plot(
                arrayx, -1*arrayy[0], label=label)
        else:
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
        axes[length_for_axis].set_xlim((time_range[0], time_range[1]))
        # axes[length_for_axis].set_xlim(axes[length_for_axis].get_xlim()[::-1])

    def graphing_ratio(label, E, B, time, times_B):
        length_for_axis = 0
        try:
            length_for_axis += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_PF_chosen"])
        except TypeError:
            pass
        if(dict['FAC'] == True):
            length_for_axis += 1
        else:
            pass
        if(dict['Difference'] == True):
            length_for_axis += 1
        else:
            pass
        try:
            length_for_axis += len(dict['sky_map_values'])
        except TypeError:
            pass
        for satellites in range(len(B)):
            for i in range(2):
                B_satellite = B[satellites][times_B]
                if(i == 0):
                    ratio = E[satellites][:, 0]/B_satellite[:, 1]  # North/East
                    label_ratio = r"$E_{Polodial}/B_{Azimuthal}$"
                else:
                    ratio = E[satellites][:, 1]/B_satellite[:, 0]  # East/North
                    label_ratio = r"$E_{Azimuthal}/B_{Polodial}$"
                ratio = ratio*10**6
                N = len(ratio)
                T = 1/16

                yf = fft(ratio)
                xf = fftfreq(N, T)[:N//2]

                if(i == 1):
                    ax21 = axes[length_for_axis].twinx()
                    ax21.plot(np.nan, np.nan)  # to make color correct
                    ax21.plot(
                        xf, 2.0/N * np.abs(yf[0:N//2]), label=r"{}".format(label_ratio))
                    ax21.ticklabel_format(
                        style='scientific', axis='y', scilimits=(0, 0))
                    ax21.set_ylabel(r"{}".format(label_ratio))
                    axes[length_for_axis].plot(
                        np.nan, np.nan, label=r"{}".format(label_ratio))

                else:
                    axes[length_for_axis].plot(
                        xf, 2.0/N * np.abs(yf[0:N//2]), label=r"{}".format(label_ratio))
                    axes[length_for_axis].set_ylabel(r"{}".format(label_ratio))

        axes[length_for_axis].set_ylabel(r"{}".format(label_ratio))
        axes[length_for_axis].legend(loc=1)
        axes[length_for_axis].ticklabel_format(
            style='scientific', axis='y', scilimits=(0, 0))

    def Graphing_skymap(pixel, time, spacecraft):
        length_for_axis = 0
        try:
            length_for_axis += len(dict["graph_E_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_B_chosen"])
        except TypeError:
            pass
        try:
            length_for_axis += len(dict["graph_PF_chosen"])
        except TypeError:
            pass
        if(dict['FAC'] == True):
            length_for_axis += 1
        else:
            pass
        if(dict['Difference'] == True):
            length_for_axis += 1
        else:
            pass
        for i in range(len(pixel)):  # length of platforms basically
            for j in range(len(pixel[0])):  # Length of satellites selected
                for k in range(len(pixel[i][j])):
                    if(pixel[i][j][k] == 0):
                        pixel[i][j][k] = np.nan
                axes[i+length_for_axis].plot(time[i], pixel[i]
                                             [j], label="".join(["swarm ", spacecraft[j]]))
            axes[i+length_for_axis].legend(loc=2)
            axes[i+length_for_axis].set_title(
                "".join([dict['sky_map_values'][i][0], dict['sky_map_values'][i][1]]))
            axes[i+length_for_axis].set_ylabel('Nearest Pixel intensity')
            axes[i+length_for_axis].set_xlim((time_range[0], time_range[1]))

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
            emph = []
            radius_total, lat_total, lon_total = [], [], []
            for i in range(len(collectionE)):
                dsE = requester(  # requests data
                    collectionE[i], measurements_flat, False, asynchronous=False, show_progress=False)

                # QUALITY MUST BE CHECKED!!

                if(len(dsE) != 0):  # checks if empty
                    # Checks if space-craft is selected
                    if("".join(["swarm", dsE["Spacecraft"][0].lower()]) in labels):
                        has_E.append(True)
                        satellites_with_E.append(
                            "".join(("swarm", dsE["Spacecraft"][0].lower())))
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

                            Bdata, Bmodel = arrangement()
                            times_of_b_for_flux = Time_corrections(
                                dsE.index.to_numpy(), dsB.index.to_numpy())  # Takes times of both E and B and finds the closest values in B to E
                            return times_of_b_for_flux, Bmodel

                        times_for_flux, MFA_vector = B_Logic_For_E()

                        radius, lattiude, longitude = dsE["Radius"].to_numpy(
                        ), dsE['Latitude'].to_numpy(), dsE['Longitude'].to_numpy()  # Gets Emphermis data
                        emph.append([radius, lattiude, longitude])
                        r_nec = Coordinate_change(lattiude, longitude, radius)
                        B

                        datamfa = MFA(ElectricNEC, MFA_vector, np.asarray(
                            r_nec).T)  # Calls MFA with (3xn) vectors

                        return_data.append(
                            datamfa)
                        returned_times = dsE.index.to_numpy()
                        if(dict["graph_E_chosen"] == None):
                            pass
                        else:

                            graphingE(
                                "".join(("Swarm ", dsE["Spacecraft"][0])), dsE.index.to_numpy(), datamfa)
                    else:
                      # Says theres no E component
                        has_E.append(False)
                else:  # Says theres no E component
                    has_E.append(False)
            return return_data, times_for_flux, returned_times, satellites_with_E

        def B():

            return_data = []
            for i in range(len(collectionB)):  # Goes through every satellite
                # Data package level, data/modes, **kwargs
                dsmodel_res = requester(collectionB[i], measurements[0], True,
                                        asynchronous=False, show_progress=False)
                if("".join(("swarm", dsmodel_res["Spacecraft"][0].lower())) in labels):
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

                    # only needs to pass data back if we need to look at pyonting flux
                    if(dict["graph_PF_chosen"] == None):
                        pass
                    else:
                        if(has_E[i] == True):
                            return_data.append(datamfa)

            else:
                pass
            return return_data

        def F(space_craft_with_E):
            data_return = []
            data_stuff = []
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
                    data_stuff.append(ds)
                else:
                    pass
            return data_return, data_stuff
        if(dict["graph_E_chosen"] == None and dict["graph_PF_chosen"] == None):
            pass
        else:
            # E field, indexs of B for time, times for plotting flux
            efield, times_for_b, time_plotting, space_craft_with_E = E()
        if(dict["graph_B_chosen"] == None and dict["graph_PF_chosen"] == None):
            pass
        else:
            bfield = B()

        if(dict["FAC"] == True):
            FAC_data, Low_hertz_dataframe = F(space_craft_with_E)

        def pontying_flux():  # Take rough estimate by doing N cross E to get C poynting flux
            nonlocal bfield, efield
            return_data = []

            for i in range(len(efield)):  # Sees how many E data points we have
                bflux = bfield[i]
                bflux = bflux[times_for_b]

                eflux = efield[i]

                flux = np.cross(eflux, bflux)
                flux_individual = np.transpose(flux)
                graphingFlux(
                    space_craft_with_E[i], time_plotting, flux_individual)
                return_data.append(
                    [space_craft_with_E[i], time_plotting, flux_individual[2]])
            return return_data

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
                        flux_time_corrected/np.max(np.abs(flux_time_corrected)), np.array(fac[i][2])/np.max(np.abs(fac[i][2]))], i)

        def skymap():
            pixel_chosen_total = []
            sat_time_each_platform = []

            for k in range(len(dict["sky_map_values"])):
                lat_satellite, lon_satellite = [], []
                asi_array_code = dict["sky_map_values"][k][0]
                location_code = dict["sky_map_values"][k][1]

                def ASI_logic():

                    alt = 110  # footprint value
                    if(asi_array_code.lower() == 'themis'):
                        asi = asilib.asi.themis(
                            location_code, time_range=time_range, alt=alt)
                        cadence = 3
                    elif(asi_array_code.lower() == 'rego'):
                        asi = asilib.asi.rego(
                            location_code, time_range=time_range, alt=alt)
                        cadence = 3
                    elif(asi_array_code.lower() == 'trex_nir'):
                        asi = asilib.asi.trex.trex_nir(
                            location_code, time_range=time_range, alt=alt)
                        cadence = 6

                    return asi, cadence
                asi, cadence = ASI_logic()
                emph, space_craft_label = empharisis_processing(cadence)

                # return cloest_time_array

                for i in range(len(emph)):  # Conjunction logic
                    # Trex is 6 second cadence compared to 3 of rego and themos

                    sat_time = np.array(emph[i][0])  # sets timestamp
                    sat_lla = np.array(
                        [emph[i][1], emph[i][2], emph[i][3]]).T

                    conjunction_obj = asilib.Conjunction(
                        asi, (sat_time, sat_lla))

                    # Converts altitude to assumed auroral height
                    # conjunction_obj.lla_footprint(map_alt=110)
                    # lat, lon, ignored = aacgmv2.convert_latlon_arr(  # Converts to magnetic coordinates
                    # in_lat=conjunction_obj.sat["lat"].to_numpy(), in_lon=conjunction_obj.sat["lon"].to_numpy(), height=alt, dtime=datetime(2021, 3, 18, 8, 0), method_code='G2A')

                    lat_satellite.append(conjunction_obj.sat["lat"].to_numpy())
                    lon_satellite.append(conjunction_obj.sat["lon"].to_numpy())

                lon, lat, ignore = _mask_low_horizon(
                    asi.skymap['lon'], asi.skymap['lat'], asi.skymap['el'], 10)

                pixel_chosen = np.zeros((len(emph), len(emph[0][0])))
                values = np.zeros((len(lon), len(lon)))
                non_blind_search = 20
                indicies_total = np.zeros(
                    (len(emph), len(emph[0][0]), 2), dtype=int)
                index_of_image = 0

                for i in range(len(emph[0][0])):  # len of time series

                    # Themis starts at time_range[0], rego and trex start a time_range[0] + a cadence
                    if(len(asi.data[0]) != len(emph[0][0]) and i == 0):
                        continue
                    elif(len(asi.data[0]) != len(emph[0][0]) and i != 0):
                        i -= 1

                    # blind optimzation, loops through time range, satellites,
                    # and lats and lons of the skymap to find the closest pixel to the satellite which can then be used to
                    # find intensities

                    if(indicies_total[0][0][0] == 0 and indicies_total[0][0][1] == 0):
                        for satellite in range(len(emph)):
                            for j in range(len(lat)):
                                for k in range(len(lon)):
                                    values[j][k] = np.sqrt(
                                        (lat[j][k]-lat_satellite[satellite][i])**2+(lon[j][k]-lon_satellite[satellite][i])**2)
                            indicies = np.unravel_index(
                                (np.abs(values)).argmin(), values.shape)
                            indicies_total[satellite, i] = indicies
                        pixel_chosen[satellite][i] = asi.data[1][index_of_image][indicies[0], indicies[1]]

                    else:  # non blind search
                        for satellite in range(len(emph)):
                            for j in range(non_blind_search):
                                for k in range(non_blind_search):
                                    values[j][k] = np.sqrt(
                                        # ugly but seperates indicies tuple
                                        (lat[indicies_total[satellite, i-1][0]-int(non_blind_search/2)+j, indicies_total[satellite, i-1][1]-int(non_blind_search/2)+k]-lat_satellite[satellite][i])**2 +
                                        (lon[indicies_total[satellite, i-1][0]-int(non_blind_search/2)+j, indicies_total[satellite, i-1][1]-int(non_blind_search/2)+k]-lon_satellite[satellite][i])**2)

                            indicies = np.unravel_index(
                                (np.abs(values)).argmin(), values.shape)

                            indicies_total[satellite, i][0] = indicies[0] + \
                                int(non_blind_search/2) + \
                                indicies_total[satellite, i-1][0]
                            indicies_total[satellite, i][1] = indicies[1] + \
                                int(non_blind_search/2) + \
                                indicies_total[satellite, i-1][0]

                        pixel_chosen[satellite][i] = asi.data[1][index_of_image][indicies[0], indicies[1]]
                    if(np.mod(i, 6) == 0 and i != 0):
                        index_of_image += 1
                    for satellite in range(len(emph)):
                        for j in range(len(lat)):
                            for k in range(len(lon)):
                                values[j][k] = np.sqrt(
                                    (lat[j][k]-lat_satellite[satellite][i])**2+(lon[j][k]-lon_satellite[satellite][i])**2)
                        indicies = np.unravel_index(
                            (np.abs(values)).argmin(), values.shape)
                        indicies_total[satellite, i] = indicies
                        pixel_chosen[satellite][i] = asi.data[1][index_of_image][indicies[0], indicies[1]]
                pixel_chosen_total.append(pixel_chosen)
                sat_time_each_platform.append(sat_time)

            return pixel_chosen_total, sat_time_each_platform, space_craft_label

        if(dict["graph_PF_chosen"] == None):
            pass
        else:
            flux = pontying_flux()
        try:
            if(['Centre'] in dict["graph_PF_chosen"] == True or dict["FAC"] == True or dict["Difference"] == True):
                Difference_plots(flux, FAC_data)
        except TypeError:
            pass
        if(dict['E_B_ratio'] == True):
            graphing_ratio(space_craft_with_E, efield,
                           bfield, time_plotting, times_for_b)

        if(len(dict["sky_map_values"]) != 0 and dict['Pixel_intensity'] == True):
            pixels, time, space_craft = skymap()
            Graphing_skymap(pixels, time, space_craft)

            # pixel_graphing(pixels, asi_array_code,location, Low_hertz_dataframe)
        else:

            pass

    requesterarraylogic()

    fig.supxlabel("Time (Day:Hour:Minute)")
    fig.suptitle("Time Versus Auroral Parameters From Swarm Spacecraft")

    # for i in range(len(axes)):  # final touches
    # mplcyberpunk.make_lines_glow(axes[i])  # adds glow to plots
    # mplcyberpunk.add_gradient_fill(
    # ax=axes[i], alpha_gradientglow=0.8, gradient_start="zero")

    return fig, axes
