from datetime import datetime, timedelta
import string
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches
import matplotlib.gridspec
import numpy as np
import pyaurorax.ephemeris as ae
asi_array_code = 'REGO'
location_code = 'FSMI'
time_range = (datetime(2021, 2, 16, 11, 47),
              datetime(2021, 2, 16, 11, 57))
area_box_km = (20, 20)
#skymap_dict = asilib.load_skymap(asi_array_code, location_code, time_range[0])

global platforms
platforms = ["swarma", "swarmb","swarmc","epop"]


def emph():

    def imager_bounds():
        global lat_bounds, lon_bounds
        lat_bounds = (skymap_dict['SITE_MAP_LATITUDE']-7,
                      skymap_dict['SITE_MAP_LATITUDE']+7)
        lon_bounds = (skymap_dict['SITE_MAP_LONGITUDE']-10,
                      skymap_dict['SITE_MAP_LONGITUDE']+10)

        return(np.array([lat_bounds, lon_bounds]))

    def coordinates_extrapolation(array):
        n = int((time_range[1] - time_range[0]
                 ).total_seconds() / 3)  # 3 second␣
        latvelocitytotal, lonvelocitytotal = [], []
        cadence = int(60/3)
        # every 3 seconds theres a time component
        time = np.array([time_range[0] + timedelta(seconds=i*3)
                         for i in range(n)])
        for i in range(len(platforms)):
            latvelocity = []
            lonvelocity = []
            for j in range(len(array[0][0])-1):
                latvelocity.append(np.linspace(
                    array[0][i][j], array[0][i][j+1], cadence))
                lonvelocity.append(np.linspace(
                    array[1][i][j], array[1][i][j+1], cadence))
            latvelocitytotal.append(np.reshape(latvelocity, -1))
            lonvelocitytotal.append(np.reshape(lonvelocity, -1))

        # def altitude finder() NEEDS IMPLEMENTING
        print("dog")
        return np.array([time, latvelocitytotal, lonvelocitytotal, (len(latvelocitytotal[0])*[500])])

    #image_bounds = imager_bounds()
    em1 = ae.search(time_range[0], time_range[1],
                    platforms=platforms)
    lattotal, lontotal = [], []
    print(em1.data[1].data_source.platform)
    for k in range(len(platforms)):
        latstarts, lonstarts = [], [],
        for i in range(len(em1.data)):  # 3 spacecraft
            if(em1.data[i].data_source.platform==platforms[k]): #sees if the data corresponds to the correct space-craft
                latstarts.append(em1.data[i].location_geo.lat)  # starting
                lonstarts.append((em1.data[i].location_geo.lon))  # ending
        lattotal.append(latstarts)
        lontotal.append(lonstarts)
    print(lattotal,lontotal)
    return coordinates_extrapolation(
        np.array([lattotal, lontotal]))


data = emph()


def main():
    array1total, array2total, array3total = [], [], []

    def footprintlogic(array, i):
        lla_110km = asilib.lla2footprint(array, 110)
        sat_azel, sat_azel_pixels = asilib.lla2azel(
            asi_array_code, location_code, time_range[0], lla_110km)
        print("pog")
        # print(array)
        print("dog")
        area_box_mask_2 = asilib.equal_area(
            asi_array_code, location_code, time_range[0], lla_110km, box_km=area_box_km
        )
        asi_brightness_2 = np.nanmean(
            image_data.images * area_box_mask_2, axis=(1, 2))
        # To play nice with plt.
        area_box_mask_2[np.isnan(area_box_mask_2)] = 0
        print("divide")

        return sat_azel_pixels, area_box_mask_2, asi_brightness_2

    def animator(array1, array2, array3):
        color = ['dodgerblue', 'purple', 'deeppink',"orange"]
        for i, (time, image, _, im) in enumerate(movie_generator):
            # Note that because we are drawing different data in each frame (a unique␣
            # image in ax[0] and the ASI time series + a guide in ax[1], we need
            # to redraw everything at every iteration.
            ax[1].clear()
            # Plot the entire satellite track, its current location, and a 20x20 km box
            # around its location.
            for j in range(len(array1)):
                ax[0].scatter(array1[j][i, 0], array1[j][i, 1],
                              c=color[j], marker='o', s=50)
                ax[0].contour(array2[j][i, :, :],
                              levels=[0.99], colors=['yellow'])
            # Plot the time series of the mean ASI intensity along the satellite path
                # ax[0] cleared by asilib.animate_fisheye_generator()
                ax[1].axvline(time, c='b')
                ax[1].plot(image_data.time, array3[j] /
                           1000, color=color[j])
                ax[1].set(xlabel='Time', ylabel=f'Mean ASI intensity\n [counts $\times␣, →10^3$]',
                          xlim=time_range)
            ax[1].text(0, 1, s="b", va='top', transform=ax[1].transAxes,
                       color='black', fontsize=20)
            # Annotate the location_code and satellite info in the top-left corner.
            ax[0].text(0, 1, '(a)', va='top', transform=ax[0].transAxes,
                       color='white', fontsize=20)
    data_animation = []

    fig, ax = plt.subplots(
        2, 1, figsize=(7, 8.5), gridspec_kw={'height_ratios': [4, 1]}
    )
    # Initiate the movie generator function. Any errors with the data will be␣

    movie_generator = asilib.animate_fisheye_generator(
        asi_array_code, location_code, time_range, azel_contours=True, ax=ax[0]
    )
    # Use the generator to get the images and time stamps to estimate mean the ASI
    # brightness along the satellite path and in a (20x20 km) box.
    image_data = movie_generator.send('data')

    for i in range(len(platforms)):
        dataforfootprint = np.array(
            [data[0], data[1][i], data[2][i], data[3]]).T
        array1, array2, array3 = footprintlogic(dataforfootprint, i)
        array1total.append(array1)
        array2total.append(array2)
        array3total.append(array3)
    animator(array1total, array2total, array3total)

    plt.subplots_adjust(wspace=0, hspace=0, right=0.98,
                        left=0.12, bottom=0.05, top=0.99)
    print(f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "animations"}')


main()
