from datetime import datetime, timedelta
import string
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches
import matplotlib.gridspec
import numpy as np
import pyaurorax.ephemeris as ae
import asilib
asi_array_code = 'TREX'
location_code = 'rabb'
time_range = (datetime(2021, 3, 18, 8, 10),
              datetime(2021, 3, 18, 8, 30))

area_box_km = (20, 20)
#skymap_dict = asilib.load_skymap(asi_array_code, location_code, time_range[0])
print(asilib.config['ASI_DATA_DIR'])
global platforms
platforms = ["swarma", "swarmb", "swarmc", "epop"]
sky_map_values = ['rego', 'fsmi']  # [['rego', 'fsmi'], ['themis', 'fsmi'],
# ['rego', 'gill'], ['trex_nir', 'rabb']]
rearth = 6378.1370


def emph():
    print(platforms, "test")

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
        print("ree")

        def altitude():
            altitude_total = []
            altepop = []
            altswarma = []
            altswarmb = []
            altvelocityepop = []
            altvelocityswarma = []
            altvelocityswarmb = []
            altswarmc = []
            altvelocityswarmc = []
            for i in range(len(platforms)):

                if(platforms[i] == "epop"):
                    for k in range(len(em1.data)):
                        if(em1.data[k].data_source.platform == "epop"):
                            altepop.append(
                                em1.data[k].metadata["radial_distance"]-rearth)
                    for j in range(len(altepop)-1):
                        altvelocityepop.append(np.linspace(
                            altepop[j], altepop[j+1], 20))
                    altvelocityepop = np.reshape(altvelocityepop, -1)
                    altitude_total.append(altvelocityepop)

                elif(platforms[i] == "swarma"):
                    for k in range(len(em1.data)):
                        if(em1.data[k].data_source.platform == "swarma"):
                            altswarma.append(
                                em1.data[k].metadata["radial_distance"]-rearth)
                    for j in range(len(altswarma)-1):
                        altvelocityswarma.append(np.linspace(
                            altswarma[j], altswarma[j+1], 20))
                    altvelocityswarma = np.reshape(altvelocityswarma, -1)
                    altitude_total.append(altvelocityswarma)

                elif(platforms[i] == "swarmb"):
                    for k in range(len(em1.data)):
                        if(em1.data[k].data_source.platform == "swarmb"):
                            altswarmb.append(
                                em1.data[k].metadata["radial_distance"]-rearth)
                    for j in range(len(altswarmb)-1):
                        altvelocityswarmb.append(np.linspace(
                            altswarmb[j], altswarmb[j+1], 20))
                    altvelocityswarmb = np.reshape(altvelocityswarmb, -1)
                    altitude_total.append(altvelocityswarmb)

                elif(platforms[i] == 'swarmc'):
                    for k in range(len(em1.data)):
                        if(em1.data[k].data_source.platform == "swarmc"):
                            altswarmc.append(
                                em1.data[k].metadata["radial_distance"]-rearth)
                    for j in range(len(altswarmc)-1):
                        altvelocityswarmc.append(np.linspace(
                            altswarmc[j], altswarmc[j+1], 20))
                    altvelocityswarmc = np.reshape(altvelocityswarmc, -1)
                    altitude_total.append(altvelocityswarmc)
                else:
                    pass

            return altitude_total

            # for i in range(len(platforms)):
            # for j in range(len(platforms)):
        altitude_array = altitude()
        return [time, latvelocitytotal, lonvelocitytotal, altitude_array]

    # image_bounds = imager_bounds()
    global em1
    em1 = ae.search(time_range[0], time_range[1],
                    platforms=platforms)
    lattotal, lontotal = [], []
    for k in range(len(platforms)):
        latstarts, lonstarts = [], [],
        for i in range(len(em1.data)):  # 3 spacecraft
            # sees if the data corresponds to the correct space-craft
            if(em1.data[i].data_source.platform == platforms[k]):
                latstarts.append(em1.data[i].location_geo.lat)  # starting
                lonstarts.append((em1.data[i].location_geo.lon))  # ending
        lattotal.append(latstarts)
        lontotal.append(lonstarts)
    return coordinates_extrapolation(
        np.array([lattotal, lontotal]))

 # [time, satellite_lattitute,.] #add everything in main


def main():
    print(__name__)
    save_file = []
    data = emph()
    # Finds the footprint of selected region with each selected spacecraft

    def animator():

        for i, (time, image, _, im) in enumerate(movie_generator):
            # Plot the entire satellite track, its current location, and a 20x20 km box
            # around its location.
            ax[1].clear()
            ax[2].clear()
            for j in range(len(sat_azel_pixels_total)):
                ax[0].plot(sat_azel_pixels_total[j][:, 0],
                           sat_azel_pixels_total[j][:, 1], 'blue')
                ax[0].scatter(sat_azel_pixels_total[j][i, 0], sat_azel_pixels_total[j][i, 1],
                              c='red', marker='o', s=50)
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

    fig, ax = plt.subplots(
        3, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [4, 1, 1]}, constrained_layout=True
    )
    sat_azel_pixels_total, area_box_mask_2_total, asi_brightness_2_total = [], [], []
    asi_array_code = sky_map_values[0]
    location_code = sky_map_values[1]
    alt = 110
    if(asi_array_code.lower() == 'themis'):
        asi = asilib.themis(
            location_code, time_range=time_range, alt=alt)
    elif(asi_array_code.lower() == 'rego'):
        asi = asilib.rego(
            location_code, time_range=time_range, alt=alt)
        print("test")
    elif(asi_array_code.lower() == 'trex_nir'):
        asi = asilib.trex_nir(
            location_code, time_range=time_range, alt=alt)
    else:
        assert "error in asi_code"
    # Initiate the movie generator function. Any errors with the data will be␣

    movie_generator = asi.animate_fisheye_gen(
        ax=ax[0], azel_contours=True, overwrite=True, cardinal_directions='NE'
    )
    # Use the generator to get the images and time stamps to estimate mean the ASI
    # brightness along the satellite path and in a (20x20 km) box.
    sat_lla_total, sat_azel_pixels_total, nearest_pixel_intensity_total, area_intensity_total, area_mask_total = [], [], [], [], []
    for i in range(len(platforms)):  # length of spacecraft

        sat_time = data[0]
        print(sat_time, i)
        sat_lla = np.array(  # lat, long, alt
            [data[1][i], data[2][i], alt * np.ones(len(data[2][i]))]).T
        conjunction_obj = asilib.Conjunction(asi, (sat_time, sat_lla))

        # Normally the satellite time stamps are not the same as the ASI.
        # You may need to call Conjunction.interp_sat() to find the LLA coordinates
        # at the ASI timestamps.
        # Map the satellite track to the imager's azimuth and elevation coordinates and
        # image pixels. NOTE: the mapping is not along the magnetic field lines! You need
        # to install IRBEM and then use conjunction.lla_footprint() before
        # calling conjunction_obj.map_azel.
        sat_azel, sat_azel_pixels = conjunction_obj.map_azel()
        nearest_pixel_intensity = conjunction_obj.intensity(box=None)
        area_intensity = conjunction_obj.intensity(box=(10, 10))
        area_mask = conjunction_obj.equal_area(box=(10, 10))

        # Need to change masked NaNs to 0s so we can plot the rectangular area contours.
        area_mask[np.where(np.isnan(area_mask))] = 0
        sat_lla_total.append(sat_lla)
        sat_azel_pixels_total.append(sat_azel_pixels)
        nearest_pixel_intensity_total.append(nearest_pixel_intensity)
        area_intensity_total.append(area_intensity)
        area_mask_total.append(area_mask)
        # sat_azel_pixels, area_box_mask_2, asi_brightness_2 for each satellite
    print(len(area_intensity_total))
    animator()

    print(
        f'Movie saved in {asilib.config["ASI_DATA_DIR"] / "animations"}', )

    movie_container = 'mp4'
    movie_address = f'{time_range[0].strftime("%Y%m%d_%H%M%S")}_' \
        f'{time_range[1].strftime("%H%M%S")}_' \
        f'{asi_array_code.lower()}_{location_code.lower()}_fisheye.{movie_container}'

    movie_address_total = asilib.config["ASI_DATA_DIR"] / \
        'animations'/movie_address
    print(movie_address_total)
    save_file.append(movie_address_total)
    return save_file


if __name__ == '__main__':  # arg parse look up
    test = main()
