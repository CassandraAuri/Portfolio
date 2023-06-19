from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
##import aacgmv2
import asilib
import asilib.asi
from Animation_GUI import emph


satellites = ["epop"]

time_range = (datetime(2021, 3, 18, 8, 14), datetime(2021, 3, 18, 8, 18))
alt = 110
satellites_selected_dict = {
    "time_range": time_range,
    "satellite_graph": ["epop"]
}

fig, ax = plt.subplots(  # intializes plots
    3, 1,  figsize=(15, 10), gridspec_kw={'height_ratios': [4, 1, 1]}, constrained_layout=True
)

asi_selected = 'trex_nir'
location_code = 'rabb'


def ASI_selection():  # Logic for choosing setting the polling rate and creates the imager class for the desired GB camera

    if(asi_selected.lower() == 'themis'):
        polling_rate = 3
        asi = asilib.asi.themis(
            location_code, time_range=time_range, alt=alt)
    elif(asi_selected.lower() == 'rego'):
        polling_rate = 3
        asi = asilib.asi.rego(
            location_code, time_range=time_range, alt=alt)
    elif(asi_selected.lower() == 'trex_nir'):
        polling_rate = 6
        asi = asilib.asi.trex.trex_nir(
            location_code, time_range=time_range, alt=alt)
    return asi, polling_rate


def satellite_logic():

    lon_sattelite = []
    lat_satelite = []
    asi, rate = ASI_selection()
    data = emph(satellites_selected_dict, rate)

    def Satellite_position(i, asi, data):
        print(len(data[1][i]), len(data[2][i]), len(data[3][i]), i)
        conj_obj = asilib.Conjunction(asi, (data[0],  np.array(  # lat_sattelite, long, alt
            [data[1][i], data[2][i], data[3][i]]).T))
        # changes coordinates to a height of 110, which assumed auroral height
        conj_obj.lla_footprint(map_alt=110)

        # lat, lon, ignored = aacgmv2.convert_latlon_arr(
        # in_lat=conj_obj.sat["lat"].to_numpy(), in_lon=conj_obj.sat["lon"].to_numpy(), height=alt, dtime=datetime(2021, 3, 18, 8, 0), method_code='G2A')
        return conj_obj.sat["lat"].to_numpy(), conj_obj.sat["lon"].to_numpy()

    print(len(satellites))
    for i in range(len(satellites)):
        lat_lon = Satellite_position(i, asi, data)
        lat_satelite.append(lat_lon[0])
        lon_sattelite.append(lat_lon[1])
    return asi, lat_satelite, lon_sattelite


asi, lat_satelite, lon_satelite = satellite_logic()
print(asi.skymap)


def _mask_low_horizon(lon_map, lat_map, el_map, min_elevation, image=None):  # from asilib
    """
        Mask the image, skymap['lon'], skymap['lat'] arrays with np.nans
        where the skymap['el'] < min_elevation or is nan.
        """
    idh = np.where(np.isnan(el_map) | (el_map < min_elevation))
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

    return x, y, image_copy


lon, lat, z = _mask_low_horizon(
    asi.skymap['lon'], asi.skymap['lat'], asi.skymap['el'], 10)
gen = asi.animate_map_gen(overwrite=True, ax=ax[0], min_elevation=10)
# magnetic_height=alt, magnetic_time=time_range[0])
##locations = [['themis', 'fsim'], ['trex_nir', 'rabb'], ['trex_nir', 'gill']]

plots = []


for i, (time, image, axes, im) in enumerate(gen):

    ax[1].clear()
    ax[2].clear()  # clears time series
    try:
        # Gets rid of satellite position (scatter plot) and cleans the satellite-track plot
        for j in range(len(to_remove_scatter)):
            to_remove_scatter[j].set_visible(False)

        for j in range(len(to_remove_plots)):
            to_remove_plots[j].pop()

    except NameError:  # If not initialized, pass
        pass

    to_remove_scatter = []
    to_remove_plots = []

    for satellite in range(len(satellites)):  # Loop through satellites
        indicies = (  # creates a tuple of the closest indicies
            np.unravel_index(
                (np.abs(lon - lon_satelite[satellite][i])).argmin(), lon.shape),
            np.unravel_index(
                (np.abs(lat - lat_satelite[satellite][i])).argmin(), lat.shape)
        )
        print(indicies)
        to_remove_scatter.append(
            ax[0].scatter(lon[indicies[0][0], indicies[0][1]], lat[indicies[1][0], indicies[1][1]], s=50, color='gold'))

        to_remove_plots.append(ax[0].plot(lon_satelite[satellite][i],  # Scatter plot for satellite position from empherasis data
                                          lat_satelite[satellite][i], color='blue'))

        to_remove_plots.append(ax[0].plot(lon_satelite[satellite],
                                          lat_satelite[satellite], color='blue'))

        # Plot the ASI intensity along the satellite path
    vline1 = ax[1].axvline(time, c='b')
    vline2 = ax[2].axvline(time, c='b')

    # Annotate the location_code and satellite info in the top-left corner.

    ax[1].set(ylabel='ASI intensity\nnearest pixel [counts]')
    ax[2].set(xlabel='Time',
              ylabel='ASI intensity\n10x10 km area [counts]')
    ax[1].set_xlim(time_range[0], time_range[1])
    ax[2].set_xlim(time_range[0], time_range[1])
