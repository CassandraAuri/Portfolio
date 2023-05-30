import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from NEC import EBplotsNEC
from MFA_EB import EBplotsMFA
from Animation_GUI import emph
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
from datetime import datetime, timedelta, date, time
import asilib


st.title("Date and conjunction time")


def Animation(timerange):
    def graphing_animation(dict, data):
        
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
        for k in range(len(dict["sky_map_values"])):  # Make function REFACTOR
            sat_azel_pixels_total, area_box_mask_2_total, asi_brightness_2_total = [], [], []
            asi_array_code = dict["sky_map_values"][k][0]
            location_code = dict["sky_map_values"][k][1]
            alt = 110
            if(asi_array_code.lower() == 'themis'):
                asi = asilib.themis(
                    location_code, time_range=time_range, alt=alt)
            elif(asi_array_code.lower() == 'rego'):
                asi = asilib.rego(
                    location_code, time_range=time_range, alt=alt)
                print("test")
            elif(asi_array_code.lower() == 'trex'):
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
            for i in range(len(platforms)):  # length of spacecraft REFACTOR

                sat_time = data[0]
                sat_lla = np.array(
                    [data[1][i], data[2][i], alt * np.ones(len(data[1][i]))]).T
                conjunction_obj = asilib.Conjunction(
                    asi, (sat_time, sat_lla))

                # Normally the satellite time stamps are not the same as the ASI.
                # You may need to call Conjunction.interp_sat() to find the LLA coordinates
                # at the ASI timestamps.
                # Map the satellite track to the imager's azimuth and elevation coordinates and
                # image pixels. NOTE: the mapping is not along the magnetic field lines! You need
                # to install IRBEM and then use conjunction.lla_footprint() before
                # calling conjunction_obj.map_azel.
                sat_azel, sat_azel_pixels = conjunction_obj.map_azel()
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
                f'{asi_array_code.lower()}_{location_code.lower()}_fisheye.{movie_container}'

            movie_address_total = asilib.config["ASI_DATA_DIR"] / \
                'animations'/movie_address
            print(movie_address_total)
            save_file.append(movie_address_total)
        return save_file

    def Animation_GUI():
        st.title("Animation Interface:")
        station_count = st.number_input(label="Number of Stations to animate", min_value=1, max_value=4, value=1, step=None,
                                        format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        def station_logic():

            count = station_count
            col = st.columns(count)
            # numpy doesnt have string arrays, only character arrays
            values = [[None, None]]*count

            def station_GUI():
                for i in range(count):
                    # initalizies our site state (project already initialized in line 41)
                    if "".join([str(i), "site"]) not in st.session_state:
                        st.session_state["".join([str(i), "site"])] = []
                    with col[i]:  # each column
                        st.multiselect("Name of Project", ["REGO", "THEMIS"],  key="".join(
                            [str(i), "project"]), max_selections=1)  # Sets the project

                        # if no project, do not execute site parameters
                        if(st.session_state["".join([str(i), "project"])] != []):
                            # if rego project, select site
                            if(st.session_state["".join([str(i), "project"])][0] == "REGO"):
                                st.multiselect("Name of Site", ["FSMI", "GILL", "RESU", "TALO"],  key="".join(
                                    [str(i), "site"]), max_selections=1)

                        # if project is themis, select themis site
                            if(st.session_state["".join([str(i), "project"])][0] == "THEMIS"):
                                st.multiselect("Name of Site", [
                                    "FSMI", "GILL", "RESU"], key="".join([str(i), "site"]), max_selections=1)

                        else:
                            pass

            def value_setter():  # sets values of selected sites and projects
                nonlocal values
                for i in range(count):  # goes through all columns
                    # doesn't add values if empty
                    if(st.session_state["".join([str(i), "project"])] == []):
                        pass

                    else:
                        # doesn't add values if empty
                        if(st.session_state["".join([str(i), "site"])] != []):
                            project = st.session_state["".join(
                                [str(i), "project"])][0]

                            sites = st.session_state["".join(
                                [str(i), "site"])][0]

                            values[i] = [project, sites]
                        else:
                            pass

            station_GUI()
            value_setter()

            return values

        skymap_values = station_logic()

        Animation_dict = {
            "time_range": timerange,
            "satellite_graph": st.session_state["Satellite_Graph"],
            "sky_map_values": skymap_values
        }

        def Animation_function_caller():
            def Animate_graph():
                # Gets the figures and axes from cache
                fig, axes = st.session_state['Graph']
                n = int((Animation_dict['time_range'][1] - Animation_dict['time_range'][0]
                         ).total_seconds() / 3)  # Finds the number of frames needed
                time = np.array([Animation_dict['time_range'][0] + timedelta(seconds=i*3)  # Creates array for x-axis (time)
                                 for i in range(n)])
                axes_changed = axes

                def Update(i):
                    # Goes through each axes and draw a vertical line at selected time to show where animation is
                    for j in range(len(axes)):

                        lines = axes_changed[j].axvline(time[i],  linewidth=1,
                                                        linestyle='dashed', color='red')

                lin_ani = animation.FuncAnimation(
                    fig, Update, frames=n)  # Creates animation
                FFwriter = animation.FFMpegWriter(fps=10)  # Writes to mp4
                lin_ani.save('animation.mp4', writer=FFwriter)
                print('success')

            data = emph(Animation_dict)
            animation_strings = graphing_animation(Animation_dict, data)
            print(animation_strings)

            try:
                clip1 = VideoFileClip(r"{}".format(animation_strings[0]))
            except IndexError:
                clip1 = None
            try:
                clip2 = VideoFileClip(r"{}".format(animation_strings[1]))
            except IndexError:
                clip2 = None
            try:
                clip3 = VideoFileClip(r"{}".format(animation_strings[2]))
            except IndexError:
                clip3 = None
            try:
                clip4 = VideoFileClip(r"{}".format(animation_strings[3]))
            except IndexError:
                clip4 = None
            print(clip1, clip2, clip3, clip4)
            if(clip3 == None and clip2 == None and clip4 == None):
                combined = clips_array([[clip1]])
            elif(clip4 == None and clip3 == None):
                combined = clips_array([[clip1, clip2]])
            elif(clip4 == None):
                combined = clips_array([[clip1, clip2, clip3]])
            else:
                combined = clips_array([[clip1, clip2], [clip3, clip4]])
            if('Graph' in st.session_state):
                Animate_graph()
                clip_graph = VideoFileClip('animation.mp4')
                if(clip3 == None and clip2 == None and clip4 == None):
                    combined = clips_array([[clip1, clip_graph]])
                elif(clip4 == None and clip3 == None):
                    combined = clips_array([[clip1, clip2, clip_graph]])
                elif(clip4 == None):
                    combined = clips_array(
                        [[clip1, clip2], [clip3, clip_graph]])
                else:
                    combined = clips_array(
                        [[clip1, clip2], [clip3, clip4], [clip_graph, clip_graph]])

            combined.write_videofile("animation_display.mp4")
            st.video("animation_display.mp4")
            st.session_state['Animation_logic_completed'] = True

        # calls to make one column but should dynamiically update
        # station_logic(1)
        button_for_animation = st.button(
            label="Render graphs", key="Animation_executer")
        if(button_for_animation == True):
            Animation_function_caller()
            print("ree-fuck-this-shitbag-fucking-hell-scape")
    Animation_GUI()


def Graph():
    st.title("Graph Interface:")

    def Graph_options_B(coord_options):
        st.multiselect(label="What directions of B would you like to graph",
                       options=coord_options, key="B_options_to_use")
        st.multiselect(label="What frequency would you like to use",
                       key="Frequency_B", options=["1Hz", "50Hz"], max_selections=1)

    def Graph_options_E(coord_options):
        st.multiselect(label="What directions of E would you like to graph",
                       options=coord_options, key="E_options_to_use")
        st.multiselect(label="What frequency would you like to use",
                       key="Frequency_E", options=["2Hz", "16Hz"], max_selections=1)

    def Graph_options_F(ignored):
        pass

    def Graph_options_PF(coord_options):
        st.multiselect(label="What directions of Ponyting Flux would you like to graph",
                       options=coord_options, key="PF_options_to_use")

    options_for_graphs = ["B", "E", "FAC", "Poynting flux"]
    Graph_functions = [Graph_options_B, Graph_options_E,
                       Graph_options_F, Graph_options_PF]

    def GUI_interface():

        graphs = st.multiselect(label="What would you like to graph",
                                options=options_for_graphs, key="Graph_select", default=None)

        coordinate_system = st.multiselect(label="What coordinate system would you like it in", options=[
                                           "North East Centre", "Mean-field aligned"], max_selections=1, key="Coordinate_system")
        return coordinate_system, graphs

    def Drop_down_menus(coordinate_system, graphs):
        try:  # tries to generate column but doesn't work if the index of the coordinate system is 0
            # sets coordinate system to give to the functions in Graph_functions
            if(coordinate_system[0] == "North East Centre"):
                coord_options = ["North", "East", "Centre"]
            elif(coordinate_system[0] == "Mean-field aligned"):
                coord_options = ["Mean-field", "Azimuthal", "Polodial"]
            try:  # st.columns doesn't like len(0)
                # creates columns for each variable to graph ie: B, E etc
                col = st.columns(len(graphs))
                for i in range(len(graphs)):  # Initializes columns
                    # 2D for loop to see if selected graph option corresponds to which Graph_function
                    for j in range(len(options_for_graphs)):
                        with col[i]:  # GUI columns
                            # if graphs options is the same index as the Graph_functions, calls graph functions
                            if(graphs[i] == options_for_graphs[j]):
                                # calls index in function array, thus calling the corresponding function with *args of the coordinate system
                                Graph_functions[j](coord_options)
            except st.errors.StreamlitAPIException:
                pass
        except IndexError:
            pass

    coordinate_system_selected, graphs_selected = GUI_interface()
    Drop_down_menus(coordinate_system_selected, graphs_selected)
    if "Difference" not in st.session_state:
        st.session_state["Difference"] = False
    st.checkbox(label="would you like to find the normalized difference between FAC and Pyonting flux (must select both FAC and pyonting flux centre)",
                key="Difference", value=st.session_state["Difference"])


def Render_Graph(timerange):
    parameters = ["Satellite_Graph", "Satellite_Graph",
                  "B_options_to_use", "Frequency_B", "E_options_to_use", "PF_options_to_use", "Frequency_E"]
    key_name = ["satellite_graph", "coordinate_system",
                "graph_B_chosen", "B_frequency", "graph_E_chosen", "graph_PF_chosen", "E_frequency"]
    for i in range(len(parameters)):
        # Initializes all the parameters
        if(parameters[i] not in st.session_state):
            st.session_state[parameters[i]] = None
    # Index error because if Graph hasn't been selected, [0] doesn't work as stated in many comments
    try:
        np.reshape([np.where(
            np.array(st.session_state["Graph_select"]) == 'FAC')], -1)[0]  # Finds if any index exists named FAC in the graphs selected
        FAC_boolean = True
    except IndexError:
        FAC_boolean = False
    # try:  # [0] doesnt work

    dict = {
        "time_range": timerange,
        "satellite_graph": st.session_state["Satellite_Graph"],
        "coordinate_system": st.session_state["Coordinate_system"],
        "graph_B_chosen": st.session_state["B_options_to_use"],
        "B_frequency": st.session_state["Frequency_B"],
        "E_frequency": st.session_state["Frequency_E"],
        "graph_E_chosen": st.session_state["E_options_to_use"],
        "graph_PF_chosen": st.session_state["PF_options_to_use"],
        "FAC": FAC_boolean,
        "Difference": st.session_state["Difference"]
    }
    print(dict["coordinate_system"])
    if(dict["coordinate_system"][0] == "North East Centre"):
        figaxes = EBplotsNEC(dict)
        st.session_state["Graph"] = figaxes

    if(dict["coordinate_system"][0] == "Mean-field aligned"):
        fig, axes = EBplotsMFA(dict)
        st.session_state["Graph"] = fig
        st.pyplot(fig)
    return


def Main():
    if 'time_start' not in st.session_state:
        st.session_state['time_start'] = time(8, 0)
    if 'time_end' not in st.session_state:
        st.session_state['time_end'] = time(8, 30)
    if 'date' not in st.session_state:
        st.session_state['date'] = date(2021, 3, 18)
    Animation_value = st.sidebar.checkbox(label="Would you like to display an auroral animation", value=False, help=None,
                                          on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Graph_value = st.sidebar.checkbox(label="Would you like to a graph", value=False, help=None,
                                      on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    date_range = st.date_input(label="Date of conjunction", key="date")
    time_rangestart = st.time_input(
        label="Time to the start of the conjunction", step=60, key="time_start", value=st.session_state['time_start'])

    time_rangeend = st.time_input(
        label="Time to the end of the conjunction", step=60, key="time_end", value=st.session_state['time_end'])
    timerange = (datetime.combine(date_range, time_rangestart),
                 datetime.combine(date_range, time_rangeend))
    st.multiselect(label="What satellites would you like to Graph", options=[
        "swarma", "swarmb", "swarmc", "epop"], key="Satellite_Graph")
    if(Graph_value == True):
        Graph()
        button = st.button(label="Render graphs", key="Graph_executer")
        if(button == True):
            Render_Graph(timerange)
            print("ree")
    if('Graph' in st.session_state):
        st.pyplot(st.session_state['Graph'][0])

    if(Animation_value == True):
        Animation(timerange)


if __name__ == '__main__':
    Main()


# fig=EBplots()


# st.pyplot(fig)
