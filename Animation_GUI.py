from datetime import datetime, timedelta
import string
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches
import matplotlib.gridspec
import numpy as np
import pyaurorax.ephemeris as ae
import asilib


rearth = 6378.1370


def emph(dict, cadence):
    time_range = dict["time_range"]
    platforms = dict["satellite_graph"]
    print(platforms)

    def coordinates_extrapolation(array):
        n = int((time_range[1] - time_range[0]
                 ).total_seconds() / cadence)  # 3 second␣
        latvelocitytotal, lonvelocitytotal = [], []

        interval = int(60/cadence)
        # every 3 seconds theres a time component
        time = np.array([time_range[0] + timedelta(seconds=i*cadence)
                         for i in range(n)])
        for i in range(len(platforms)):  # goes through each platform
            latvelocity = []
            lonvelocity = []
            # length is the length of the lattitude for EACH platform shape is (2 (lat,lon), len(platform),_)
            for j in range(len(array[0][i])-1):
                print(len(array[0][i])-1)
                print(i, j)
                latvelocity.append(np.linspace(
                    array[0][i][j], array[0][i][j+1], interval))
                lonvelocity.append(np.linspace(
                    array[1][i][j], array[1][i][j+1], interval))
            latvelocitytotal.append(np.reshape(latvelocity, -1))
            lonvelocitytotal.append(np.reshape(lonvelocity, -1))
        print("ree")

        def altitude():
            altitude_total = []
            for i in range(len(platforms)):
                altvelocityepop = []
                altepop = []
                for k in range(len(em1.data)):
                    print(em1.data[k].data_source.platform, platforms[i])
                    if(em1.data[k].data_source.platform == platforms[i]):
                        altepop.append(
                            em1.data[k].metadata["radial_distance"]-rearth)
                for j in range(len(altepop)-1):
                    altvelocityepop.append(np.linspace(
                        altepop[j], altepop[j+1], interval))

                print(len(altvelocityepop))
                if(len(altvelocityepop) != 0):
                    altvelocityepop = np.reshape(altvelocityepop, -1)
                    altitude_total.append(altvelocityepop)

                    print(len(altvelocityepop))

            return altitude_total

            # for i in range(len(platforms)):
            # for j in range(len(platforms)):
        altitude_array = altitude()

        return [time, latvelocitytotal, lonvelocitytotal, altitude_array]

        # for i in range(len(platforms)):
        # for j in range(len(platforms)):

    # image_bounds = imager_bounds()
    global em1
    em1 = ae.search(time_range[0], time_range[1],
                    platforms=platforms)
    lattotal, lontotal = [], []
    for k in range(len(platforms)):
        latstarts, lonstarts = [], [],
        for i in range(len(em1.data)): # 3 spacecraft
            # sees if the data corresponds to the correct space-craft
            if(em1.data[i].data_source.platform == platforms[k]):
                latstarts.append(em1.data[i].location_geo.lat)  # starting
                lonstarts.append((em1.data[i].location_geo.lon))  # ending
        lattotal.append(latstarts)
        lontotal.append(lonstarts)
    return coordinates_extrapolation(
        [lattotal, lontotal])


def graphing_animation(dict):

    data_3, data_6 = emph(dict, 3), emph(dict, 6)
    time_range = dict["time_range"]
    platforms = dict["satellite_graph"]
    save_file = []

       # Finds the footprint of selected region with each selected spacecraft

    def animator():

        for i, (time, image, _, im) in enumerate(movie_generator):
            # Plot the entire satellite track, its current location, and a 20x20 km box
            # around its location.

            ax[1].clear()
            ax[2].clear()
            """
            for j in range(len(sat_azel_pixels_total)):
                ax[0].plot(sat_azel_pixels_total[j][:, 0],
                            sat_azel_pixels_total[j][:, 1], 'blue')
                ax[0].scatter(sat_azel_pixels_total[j][i, 0], sat_azel_pixels_total[j][i, 1],
                                marker='o', s=50)
                ax[0].contour(area_mask_total[j][i, :, :],
                                levels=[0.99], colors=['yellow'])

                ax[1].plot(sat_time, nearest_pixel_intensity_total[j])
                ax[2].plot(sat_time, area_intensity_total[j])

                # Plot the ASI intensity along the satellite path
            vline1 = ax[1].axvline(time, c='b')
            vline2 = ax[2].axvline(time, c='b')

            # Annotate the location_code and satellite info in the top-left corner.
            location_code_str = (
                f'{asi_array_code}/{location_code} '
                f'LLA=({asi.meta["lat"]:.2f}, '
                f'{asi.meta["lon"]:.2f}, {asi.meta["alt"]:.2f})'
            )
            text_obj = ax[0].text(
                0,
                1,
                location_code_str,
                va='top',
                transform=ax[0].transAxes,
                color='red',
            )
            ax[1].set(ylabel='ASI intensity\nnearest pixel [counts]')
            ax[2].set(xlabel='Time',
                        ylabel='ASI intensity\n10x10 km area [counts]')
            ax[1].set_xlim(time_range[0], time_range[1])
            ax[2].set_xlim(time_range[0], time_range[1])
            """
    fig, ax = plt.subplots(  # intializes plots
            3, 1,  figsize=(15, 10), gridspec_kw={'height_ratios': [4, 1, 1]}, constrained_layout=True
        )
        # Loops through the number of stations selected by the user
    for k in range(len(dict["sky_map_values"])):
        asi_array_code = dict["sky_map_values"][k][0]
        location_code = dict["sky_map_values"][k][1]

        def ASI_logic():

            alt = 110  # footprint value
            if(asi_array_code.lower() == 'themis'):
                frame_rate = 10
                asi = asilib.asi.themis(
                    location_code, time_range=time_range, alt=alt)
                movie_generator = asi.animate_map_gen(  # initaliziation
                    ax=ax[0], azel_contours=True, overwrite=True, ffmpeg_params={'framerate': frame_rate}, magnetic_height=alt, magnetic_time=time_range[0])
            elif(asi_array_code.lower() == 'rego'):
                frame_rate = 10
                asi = asilib.asi.rego(
                    location_code, time_range=time_range, alt=alt)
                movie_generator = asi.animate_map_gen(  # initaliziation
                    ax=ax[0], azel_contours=True, overwrite=True, color_bounds=[300, 550], ffmpeg_params={'framerate': frame_rate}, magnetic_height=alt, magnetic_time=time_range[0])
                print("test")
            elif(asi_array_code.lower() == 'trex_nir'):
                frame_rate = 5
                asi = asilib.asi.trex.trex_nir(
                    location_code, time_range=time_range, alt=alt)
                movie_generator = asi.animate_map_gen(  # initaliziation
                    ax=ax[0], azel_contours=True, overwrite=True, color_bounds=[400, 700], ffmpeg_params={'framerate': frame_rate},magnetic_height=alt,magnetic_time=time_range[0])
            return asi, movie_generator
        asi, movie_generator = ASI_logic()

        # Initiate the movie generator function. Any errors with the data will be␣

            # Use the generator to get the images and time stamps to estimate mean the ASI
            # brightness along the satellite path and in a (20x20 km) box.
        alt = 110
        sat_lla_total, sat_azel_pixels_total, nearest_pixel_intensity_total, area_intensity_total, area_mask_total = [], [], [], [], []
        for i in range(len(platforms)):  # length of spacecraft REFACTOR

            # Trex is 6 second cadence compared to 3 of rego and themos
            if(asi_array_code.lower() == 'trex_nir'):
                data = data_6
            else:
                data = data_3

            sat_time = data[0]  # sets timestamp
            print(len(data[1][i]), len(data[3][i]))
            sat_lla = np.array(
                [data[1][i], data[2][i], data[3][i]]).T

            conjunction_obj = asilib.Conjunction(
                asi, (sat_time, sat_lla))

            # Normally the satellite time stamps are not the same as the ASI.
            # You may need to call Conjunction.interp_sat() to find the LLA coordinates
            # at the ASI timestamps.
            # Map the satellite track to the imager's azimuth and elevation coordinates and
            # image pixels. NOTE: the mapping is not along the magnetic field lines! You need
            # to install IRBEM and then use conjunction.lla_footprint() before
            # calling conjunction_obj.map_azel.
            conjunction_obj.lla_footprint(map_alt=110)
            sat_azel, sat_azel_pixels = conjunction_obj.map_azel()  # See ASILIB documentation
            print(__name__)
            nearest_pixel_intensity = conjunction_obj.intensity(
                box=None)
            area_intensity = conjunction_obj.intensity(box=(10, 10))
            area_mask = conjunction_obj.equal_area(box=(10, 10))

            # Need to change masked NaNs to 0s so we can plot the rectangular area contours.
            area_mask[np.where(np.isnan(area_mask))] = 0
            sat_lla_total.append(sat_lla)
            sat_azel_pixels_total.append(sat_azel_pixels)
            nearest_pixel_intensity_total.append(
                nearest_pixel_intensity)
            area_intensity_total.append(area_intensity)
            area_mask_total.append(area_mask)
        # sat_azel_pixels, area_box_mask_2, asi_brightness_2 for each satellite
        animator()

        print(
            f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "animations"}')

        movie_container = 'mp4'
        movie_address = f'{time_range[0].strftime("%Y%m%d_%H%M%S")}_' \
            f'{time_range[1].strftime("%H%M%S")}_' \
            f'{asi_array_code.lower()}_{location_code.lower()}_fisheye.{movie_container}'  # file address of movie saved by asilib

        movie_address_total = asilib.config["ASI_DATA_DIR"] / \
            'animations'/movie_address  # full address from C:
        print(movie_address_total)
        # Saves address so movie.py can load it in the GUI
        save_file.append(movie_address_total)
    return save_file
