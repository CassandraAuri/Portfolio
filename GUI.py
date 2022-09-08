from email import message
from msilib.schema import CheckBox
from requests import options
from scipy.stats import norm
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import PillowWriter
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import skew, kurtosis
from matplotlib import animation
import scipy.interpolate
from matplotlib.gridspec import GridSpec
import scipy
from scipy.stats import skew, kurtosis
from scipy.stats import norm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
import matplotlib.colors as mcolors
import matplotlib.cm as cm
root = Tk()
root.geometry('1920x1080')


def heatTimeSetting():
    # convert pandas dataframe to simple python list
    column_list = frame.columns.tolist()

    OPTIONS = column_list  # this is what solved my problem

    x_axis_var = StringVar(heatimesetting)
    Label(heatimesetting, text="Variable of x-axis").grid(row=0, column=0)
    OptionMenu(heatimesetting, x_axis_var,
               *OPTIONS).grid(row=1, column=0)

    Label(heatimesetting, text="Cross-sectional number for x-axis (must be between x min and max)").grid(row=2, column=0)
    x_choose = Entry(heatimesetting, width=5)
    x_choose.grid(row=3, column=0)

    Label(heatimesetting, text="x-axis min").grid(row=4, column=0)
    xmin = Entry(heatimesetting, width=5)
    xmin.grid(row=5, column=0)

    Label(heatimesetting, text="x-axis max").grid(row=7, column=0)
    xmax = Entry(heatimesetting, width=5)
    xmax.grid(row=8, column=0)

    Label(heatimesetting, text="x unit").grid(row=9, column=0)
    xunit = Entry(heatimesetting, width=15)
    xunit.grid(row=10, column=0)

    Label(heatimesetting, text="x unit factor(ex: cm->m= 100)").grid(row=11, column=0)
    xfac = Entry(heatimesetting, width=5)
    xfac.grid(row=12, column=0)
    xfac.insert(END, 1)
    y_axis_var = StringVar(heatimesetting)
    Label(heatimesetting, text="Variable of heatmap y-axis").grid(row=0, column=1)
    OptionMenu(heatimesetting, y_axis_var,
               *OPTIONS).grid(row=1, column=1)

    Label(heatimesetting,
          text="Cross-sectional number for heatmap y-axis").grid(row=2, column=1)
    y_choose = Entry(heatimesetting, width=5)
    y_choose.grid(row=3, column=1)

    Label(heatimesetting, text=" heatmap ymin").grid(row=4, column=1)
    ymin = Entry(heatimesetting, width=5)
    ymin.grid(row=6, column=1)

    Label(heatimesetting, text="ymax").grid(row=7, column=1)
    ymax = Entry(heatimesetting, width=5)
    ymax.grid(row=8, column=1)

    Label(heatimesetting, text="y unit").grid(row=9, column=1)
    yunit = Entry(heatimesetting, width=15)
    yunit.grid(row=10, column=1)

    Label(heatimesetting, text="y unit factor(ex: cm->m= 100)").grid(row=11, column=1)
    yfac = Entry(heatimesetting, width=5)
    yfac.grid(row=12, column=1)

    Label(heatimesetting, text="Held Variable").grid(
        row=0, column=2)  # Thing we will be scrolling through
    held_var = StringVar(heatimesetting)
    OptionMenu(heatimesetting, held_var, *OPTIONS).grid(
        row=1, column=2)

    Label(heatimesetting, text="Held Variable Value").grid(
        row=2, column=2)  # For scroll function
    held_choose = Entry(heatimesetting, width=5)
    held_choose.grid(row=3, column=2)

    Label(heatimesetting, text="Held Variable Min").grid(
        row=4, column=2)  # For scroll function
    held_choosemin = Entry(heatimesetting, width=5)
    held_choosemin.grid(row=5, column=2)

    Label(heatimesetting, text="Held Variable Max").grid(
        row=6, column=2)  # For scroll function
    held_choosemax = Entry(heatimesetting, width=5)
    held_choosemax.grid(row=7, column=2)

    Label(heatimesetting, text="held unit").grid(row=8, column=2)
    heldunit = Entry(heatimesetting, width=15)
    heldunit.grid(row=9, column=2)

    Label(heatimesetting, text="held unit factor(ex: cm->m= 100)").grid(row=10, column=2)
    heldfac = Entry(heatimesetting, width=5)
    heldfac.grid(row=11, column=2)

    try:
        ind = OPTIONS.index('x')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('X')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('Y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('Time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('T')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('t')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    if(parmst == True):
        held_choosemin.insert(END, tminn)
        held_choosemax.insert(END, tmaxn)
    heldfac.insert(END, 1)
    if(parmsx == True):
        xmin.insert(END, xminn)
        xmax.insert(END, xmaxn)
    xfac.insert(END, 1)
    if(parmsy == True):
        ymin.insert(END, yminn)
        ymax.insert(END, ymaxn)
        yfac.insert(END, 1)

    Label(heatimesetting, text="Data Variable1").grid(
        row=0, column=5)  # For scroll function
    datavar1 = StringVar(heatimesetting)
    OptionMenu(heatimesetting, datavar1, *OPTIONS).grid(row=1, column=5)

    Label(heatimesetting, text="Data Constant1(should be DC)").grid(
        row=2, column=5)  # For scroll function
    dataconstant1 = StringVar(heatimesetting)
    OptionMenu(heatimesetting, dataconstant1, *OPTIONS).grid(row=3, column=5)

    Label(heatimesetting, text="Data Unit 1").grid(row=4, column=5)
    dataunit1 = Entry(heatimesetting, width=15)
    dataunit1.grid(row=5, column=5)

    Label(heatimesetting, text="Data unit 1 factor(ex: cm->m= 100)").grid(row=6, column=5)
    dataunit1fac = Entry(heatimesetting, width=5)
    dataunit1fac.grid(row=7, column=5)

    Label(heatimesetting, text="Data Variable2").grid(
        row=8, column=5)  # For scroll function
    datavar2 = StringVar(heatimesetting)
    OptionMenu(heatimesetting, datavar2, *OPTIONS).grid(row=9, column=5)

    Label(heatimesetting, text="Data Constant2(should be DC)").grid(
        row=10, column=5)  # For scroll function
    dataconstant2 = StringVar(heatimesetting)
    OptionMenu(heatimesetting, dataconstant2, *OPTIONS).grid(row=11, column=5)

    Label(heatimesetting, text="Data Unit 2").grid(row=12, column=5)
    dataunit2 = Entry(heatimesetting, width=15)
    dataunit2.grid(row=13, column=5)

    Label(heatimesetting,
          text="Data unit 2 factor(ex: cm->m= 100)").grid(row=14, column=5)
    dataunit2fac = Entry(heatimesetting, width=5)
    dataunit2fac.grid(row=15, column=5)

    Label(heatimesetting, text="If multiple trials, which trial").grid(
        row=0, column=6)  # For scroll function
    trial = Entry(heatimesetting, width=5)
    trial.grid(row=1, column=6)

    Label(heatimesetting, text="Resolution").grid(
        row=2, column=6)
    deltat = Entry(heatimesetting, width=5)
    deltat.grid(row=3, column=6)
    vector = BooleanVar(heatimesetting)
    Checkbutton(
        heatimesetting, text="Would you like enable a vector map", variable=vector).grid(row=4, column=6)

    for i in range(len(OPTIONS)):
        maxmin = Label(heatimesetting, text=(OPTIONS[i]+" "+"Has a max of"+" "+str(np.around((np.max(frame[OPTIONS[i]])), 4)) +
                                             " "+"and a min of"+" "+str(np.around(np.min(frame[OPTIONS[i]]), 4))))
        maxmin.grid(row=i, column=4)
    Button(heatimesetting, text="Graph", command=lambda: [
        HeattimeCreator(),
        HeattimeGrapher(x_axis_var.get(), y_axis_var.get(), held_var.get(), xmin.get(), ymin.get(), xmax.get(), ymax.get(), deltat.get(),
                        x_choose.get(), y_choose.get(), held_choose.get(), held_choosemax.get(),
                        held_choosemin.get(), datavar1.get(), trial.get(), vector.get(), xunit.get(
        ), xfac.get(), yunit.get(), yfac.get(), heldunit.get(), heldfac.get(), True,
            dataconstant1.get(), datavar2.get(), dataconstant2.get(), 0, 0, 0, 0, dataunit1.get(), dataunit1fac.get(), dataunit2.get(), dataunit2fac.get(), False),  # False is just making sure scaling isnt exponential
        # ErrorHeat(xmin.get(), ymin.get(), xmax.get(), ymax.get(), x_choose.get(), y_choose.get(), held_choose.get(), held_choosemax.get(), held_choosemin.get()
    ]).grid(column=9, row=30)
    Button(heatimesetting, text="Quit", command=heatimesetting.destroy).grid(
        column=9, row=31)


def HeattimeCreator():
    global timeheat
    timeheat = Toplevel()
    timeheat.title("Heat map")
    timeheat.geometry('1500x1000')
    return timeheat


def HeattimeGrapher(xaxis, yaxis, topaxis, xmin, ymin, xmax, ymax, deltat, x_choose, y_choose,
                    heldchoose, heldmax, heldmin, datavar1, shot, vector, xunit, xfac,
                    yunit, yfac, heldunit, heldfac, firstrender, constantvar1, datavar2, constantvar2, x, y, t, dataarray, varunit1, varunit1fac, varunit2, varunit2fac, displacement):

    try:
        xmin, ymin, xmax, ymax, deltat, x_choose, y_choose, heldchoose, heldmax, heldmin = float(xmin), float(ymin), float(xmax), float(ymax), float(deltat), float(
            x_choose), float(y_choose), float(heldchoose), float(heldmax), float(heldmin)
        nplots = [[datavar1, constantvar1, varunit1, varunit1fac],
                  [datavar2, constantvar2, varunit2, varunit2fac]]
        plt.rcParams.update({'font.size': 8})
        if(firstrender == True):
            def Scaling():
                try:
                    nonlocal xmin, xmax, ymin, ymax, heldmax, heldmin, y_choose, x_choose, heldchoose
                    xmin = xmin*float(xfac)
                    xmax = xmax*float(xfac)
                    ymin = ymin*float(yfac)
                    ymax = ymax*float(yfac)
                    heldmax = heldmax*float(heldfac)
                    heldmin = heldmin*float(heldfac)
                    y_choose = y_choose*float(yfac)
                    x_choose = x_choose*float(xfac)
                    heldchoose = heldchoose*float(heldfac)
                except ValueError:
                    timeheat.destroy()
                    messagebox.showerror(
                        message="Make sure your factors are floats")

            def Varassigning():
                try:
                    nonlocal x, y, t, dataarray
                    x = frame[[xaxis]].to_numpy()
                    x = x[~np.isnan(x)]
                    y = frame[[yaxis]].to_numpy()
                    y = y[~np.isnan(y)]
                    t = frame[[topaxis]].to_numpy()
                    t = t[~np.isnan(t)]

                    x = x*float(xfac)
                    y = y*float(yfac)
                    t = t*float(heldfac)

                    dataarray = []
                    for a in range(len(nplots)):
                        data = frame[[nplots[a][0]]].to_numpy()
                        data = data[~np.isnan(data)]
                        dataconstant = frame[[nplots[a][1]]].to_numpy()
                        dataconstant = dataconstant[~np.isnan(
                            dataconstant)]
                        data = data*float(nplots[a][3])
                        dataconstant = dataconstant*float(nplots[a][3])
                        dataarray.append([data, dataconstant])
                except KeyError:
                    timeheat.destroy
                    messagebox.showerror(
                        message="Make sure every variable is assigned")

            def LatexSymbols():
                nonlocal xunit, yunit, heldunit
                xunit = "$"+xunit+"$"
                yunit = "$"+yunit+"$"
                heldunit = "$"+heldunit+"$"
            Varassigning()
            LatexSymbols()
            Scaling()
        else:
            pass
        Nt = len(t)
        Nx = len(x)
        Ny = len(y)

        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(2, 4, fig, wspace=0.65, hspace=0.4)

        def centre(zheat):
            nonlocal x, y, x_choose, y_choose, xmin, ymin, xmax, ymax
            center = np.mean(np.argwhere(zheat > 0.9), axis=0)
            xc, yc = x[int(np.around(center[0], 0))
                       ]-0.1, y[int(np.around((center[1]), 0))]
            x = x-xc
            xmin = xmin-xc
            xmax = xmax-xc
            x_choose = x_choose-xc
            y = y-yc
            ymin = ymin-yc
            ymax = ymax-yc
            y_choose = y_choose-yc

        def usercentre():
            nonlocal x, y, x_choose, y_choose, xmin, ymin, xmax, ymax
            xc = x_choose
            yc = y_choose
            x = x-xc
            xmin = xmin-xc
            xmax = xmax-xc
            x_choose = x_choose-xc
            y = y-yc
            ymin = ymin-yc
            ymax = ymax-yc
            y_choose = y_choose-yc
            print('resolved')

        def CreatePlot(y_choose, x_choose):

            def radius():
                ind1 = []
                r = []

                r_choose = np.sqrt(x_choose**2+y_choose**2)
                print(str(x_choose)+"xchoose and" +
                      str(y_choose)+"ychoose"+str(r_choose)+"rchoose")
                for i in range(Nx):
                    for j in range(Ny):
                        r.append(
                            [np.abs(np.sqrt(x[i]**2+y[j]**2)-float(r_choose)), i, j])  # gives all radi and their postions in the x y arrays

                for i in range(len(r)):
                    ind1.append((r[i][0]))  # appends the radii alone
                ind = np.argmin(ind1)  # fins the smallest radii
                # finds corresponding i and j values
                r1, r2 = r[ind][1], r[ind][2]
                return r1, r2
            r1, r2 = radius()
            for b in range(len(dataarray)):
                dataconstant = dataarray[b][1]
                data = dataarray[b][0]

                try:
                    ztimeseries = np.reshape(
                        data, (Nt, Nx, Ny, Nshot), order="F")
                except ValueError:
                    timeheat.destroy
                    messagebox.showerror(
                        message="Make sure your data variable is actually a data variable")

                try:
                    zconstant = np.reshape(dataconstant, (Nt, Nx, Ny, Nshot))
                except ValueError:
                    timeheat.destroy()
                    messagebox.showerror(
                        message="Make sure your constant data variable is actually a data variable")
                zreal = []
                heldinx = np.abs(t-heldchoose).argmin()

                if(Nt > Nx and Nt > Ny):
                    zheat = np.reshape(
                        dataconstant, (Nt, Nx, Ny, Nshot), order="F")
                elif(Nt >= Nx and Ny > Nt):
                    zheat = np.reshape(
                        dataconstant, (Ny, Nt, Nx, Nshot), order="F")
                else:
                    zheat = np.reshape(
                        dataconstant, (Nx, Nt, Ny, Nshot), order="F")
                if(b == 0 and firstrender == True):
                    centre(zheat[0, :, :, int(shot)-1])
                if(b == 0 and displacement == True):
                    usercentre()
                zheat = zheat[heldinx, :, :, int(shot)-1]

                def shotcompile():
                    zcon = []
                    ztime = []
                    for i in range(Nshot):
                        zcon.append(zconstant[:, r1, r2, i])
                        ztime.append(ztimeseries[:, r1, r2, i])
                    return zcon, ztime
                zconstant, ztimeseries = shotcompile()
                print(r1, r2)
                print(r1, r2)
                print(r1, r2)
                print(r1, r2)
                print(r1, r2)
                print(r1, r2)
                print(r1, r2)

                def c3def():
                    tmod = []
                    zreal = []
                    zmod = []
                    for j in range(Nshot):
                        # This creates so we can look at certain regions and then do analysis on each, Normally you could just render with limits, but since we analysis we need only specific region data
                        for i in range(len(t)):
                            if(t[i] >= float(heldmin) and t[i] <= float(heldmax)):
                                if(j == int(shot)-1):
                                    tmod.append(t[i])
                                    zmod.append(
                                        (ztimeseries[j][i])/zconstant[j][i])
                                zreal.append(
                                    (ztimeseries[j][i])/zconstant[j][i])
                    c3 = fig.add_subplot(gs[b, 1])
                    plt.plot(tmod, zmod, color='black', label=np.around(
                        np.sqrt(x[r1]**2+y[r2]**2), 4))
                    print(r1, r2)
                    return c3, np.reshape(zreal, -1)

                def c2def(zreal):
                    c2 = fig.add_subplot(gs[b, 2])
                    sigma = np.std((zreal))
                    mu = np.mean((zreal))
                    #logmu = 10**(sigma+mu**2/2)
                    #logsigma = np.sqrt((10**(sigma**2)-1)*logmu*2)
                    zreal = zreal/sigma
                    #mean, var, skew, kurt = norm.stats(moments='mvsk')
                    xline = np.linspace(np.min(zreal), np.max(zreal), 300)
                    plt.plot(xline, norm.pdf(xline, mu, sigma),
                             'r--', lw=4, alpha=1)
                    count, bins, ignored = plt.hist(
                        zreal, density=True, bins=100, histtype='step', alpha=0.8, lw=2, color='black', label=("skew"+"="+str(np.around(skew(zreal), 4))+" "+"and"+" "+"kurtosis" +
                                                                                                               "="+str(np.around(kurtosis(zreal, fisher=True), 4))))
                    plt.legend()
                    count = count[count != 0]
                    return c2, count

                def c1def():
                    c1 = fig.add_subplot(gs[b, 0])
                    if(vector == True):
                        [xgrad, ygrad] = np.gradient(zheat)
                        plt.quiver(x, y, xgrad, ygrad, width=0.01)
                    else:
                        plt.pcolormesh(x, y, zheat, shading="gouraud",
                                       cmap='hot')  # heatmap
                        cbar = plt.colorbar()
                        cbar.ax.set_title(
                            str(nplots[b][1])+"($"+nplots[b][2]+"$)")

                # # Constants that show where we are on heatmap
                    plt.plot((x),
                             [y_choose]*len(y), color="black")
                    # # X/Y Slices, float array is simply making a line at the chosen value with the correct length(flat line on graph)
                    plt.plot([x_choose]*len(x), (y), color="black")
                    return c1

                def GraphSettings(c1, c2, c3, count):
                    c1.set(xlim=(xmin, xmax))
                    c1.set(ylim=(ymin, ymax))
                    c1.set(xlabel=xaxis+"("+xunit+")")
                    c1.set(ylabel=yaxis+"("+yunit+")")
                    c1.set(title="shot:"+shot+'@'+topaxis+"("+heldunit+")")

                    c3.set(xlabel=topaxis+"("+heldunit+")")
                    c3.set(
                        ylabel=("$\\frac{\\delta"+" "+nplots[b][0]+"}{"+nplots[b][1]+"}"+"$"+" "+"(Arb. Units)"))
                    c3.set(title="snapshot of"+shot)
                    c2.set(
                        xlabel="$\\frac{\\delta"+" "+nplots[b][0]+"}{"+nplots[b][1]+"\\times \\sigma"+"}"+"$")
                    c2.set(ylabel="log10[PDF]")
                    c2.set(title="lognormal PDF of all trials compiled")
                    c2.set(yscale="log")
                    c2.set(ylim=(np.min(count), 1))
                c3, zreal = c3def()

                c2, count = c2def(zreal)
                c1 = c1def()
                GraphSettings(c1, c2, c3, count)
            return fig
    except ValueError:
        timeheat.destroy
        messagebox.showerror(
            message="Please make sure every value is inputted")

    fig1 = CreatePlot(y_choose, x_choose)

    def AnimationX():
        skewkurtarray = []

        def CreateSkewKurt(index, skewkurtarray):
            if(index == 0):
                global skew0x, kurt0x, skew1x, kurt1x
                skew0x = []
                kurt0x = []
                skew1x = []
                kurt1x = []

                for i in range(len(y)):
                    def radius():
                        ind1 = []
                        r = []

                        r_choose = np.sqrt(
                            x_choose**2+((ymin)+(ymax-ymin)/21*(i+1))**2)

                        for j in range(len(x)):
                            for k in range(len(y)):
                                r.append(
                                    [np.abs(np.sqrt(x[j]**2+y[k]**2)-float(r_choose)), j, k])  # gives all radi and their postions in the x y arrays

                        for j in range(len(r)):
                            ind1.append((r[j][0]))  # appends the radii alone

                        ind = np.argmin(ind1)  # fins the smallest radii
                        # finds corresponding i and j values
                        r1, r2 = r[ind][1], r[ind][2]
                        return r1, r2
                    r1, r2 = radius()
                    for a in range(len(dataarray)):
                        data = dataarray[a][0]
                        dataconstant = dataarray[a][1]
                        ztimeseries = np.reshape(
                            data, (Nt, Nx, Ny, Nshot), order="F")

                        zconstant = np.reshape(
                            dataconstant, (Nt, Nx, Ny, Nshot), order="F")

                        def shotcompile():
                            zcon = []
                            ztime = []
                            for i in range(Nshot):
                                zcon.append(zconstant[:, r1, r2, i])
                                ztime.append(ztimeseries[:, r1, r2, i])
                            return zcon, ztime
                        zconstant, ztimeseries = shotcompile()
                        tmod = []
                        zreal = []
                        # This creates so we can look at certain regions and then do analysis on each, Normally you could just render with limits, but since we analysis we need only specific region data
                        for j in range(Nshot):
                            for i in range(len(t)):
                                if(t[i] >= float(heldmin) and t[i] <= float(heldmax)):
                                    zreal.append(
                                        (ztimeseries[j][i])/zconstant[j][i])
                        skews = skew(zreal/np.std(zreal))
                        kurts = kurtosis(zreal/np.std(zreal), fisher=True)
                        skewkurtarray.append([skews, kurts, a])

                for b in range(len(skewkurtarray)):
                    if(skewkurtarray[b][2] == 0):
                        skew0x.append(skewkurtarray[b][0])
                        kurt0x.append(skewkurtarray[b][1])
                    if(skewkurtarray[b][2] == 1):
                        skew1x.append(skewkurtarray[b][0])
                        kurt1x.append(skewkurtarray[b][1])
            else:
                pass
            c4 = fig.add_subplot(gs[0, 3])
            plt.plot(skew0x, kurt0x, 'o')
            p = np.polyfit(skew0x, kurt0x, 2)
            skewline0 = np.linspace(
                np.min(skew0x), np.max(skew0x), len(kurt0x))
            yfit0 = np.polyval(p, skewline0)
            plt.plot(skewline0, yfit0, label=('a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                              str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3))))
            plt.legend()
            #plt.errorbar(skew0x, kurt0x,res, fmt='none')
            c5 = fig.add_subplot(gs[1, 3])
            plt.plot(skew1x, kurt1x, 'o')
            p = np.polyfit(skew1x, kurt1x, 2)
            skewline1 = np.linspace(
                np.min(skew1x), np.max(skew1x), len(kurt1x))
            yfit1 = np.polyval(p, skewline1)
            plt.plot(skewline1, yfit1, label=('a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                              str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3))))
            plt.legend()
            #plt.errorbar(skew1x, kurt1x, res, fmt='none')

            def GraphSetting():
                c4.set(xlabel="Skew")
                c4.set(ylabel="Kurtosis")

                c5.set(xlabel="Skew")
                c5.set(ylabel="Kurtosis")
            GraphSetting()

        def animations(i):
            plt.clf()
            ychoose = ymin+(ymax-ymin)*(i+1)/21
            print(str(ychoose)+"ychoose in animations")
            CreateSkewKurt(i, skewkurtarray)
            CreatePlot(ychoose, x_choose)
        anim = animation.FuncAnimation(
            fig, animations, frames=len(y), repeat=False)
        anim.save('Animation.gif', writer=PillowWriter(fps=3))
        AnimationDisp(len(y))

    def AnimationY():
        skewkurtarray = []

        def CreateSkewKurt(index, skewkurtarray):
            if(index == 0):
                global skew0x, kurt0x, skew1x, kurt1x
                skew0x = []
                kurt0x = []
                skew1x = []
                kurt1x = []

                for i in range(len(y)):
                    def radius():
                        ind1 = []
                        r = []

                        r_choose = np.sqrt(
                            ((xmin)+(xmax-xmin)/21*(i+1))**2+y_choose**2)

                        for j in range(len(x)):
                            for k in range(len(y)):
                                r.append(
                                    [np.abs(np.sqrt(x[j]**2+y[k]**2)-float(r_choose)), j, k])  # gives all radi and their postions in the x y arrays

                        for j in range(len(r)):
                            ind1.append((r[j][0]))  # appends the radii alone

                        ind = np.argmin(ind1)  # fins the smallest radii
                        # finds corresponding i and j values
                        r1, r2 = r[ind][1], r[ind][2]
                        return r1, r2
                    r1, r2 = radius()
                    for a in range(len(dataarray)):
                        data = dataarray[a][0]
                        dataconstant = dataarray[a][1]
                        ztimeseries = np.reshape(
                            data, (Nt, Nx, Ny, Nshot), order="F")

                        zconstant = np.reshape(
                            dataconstant, (Nt, Nx, Ny, Nshot), order="F")

                        def shotcompile():
                            zcon = []
                            ztime = []
                            for i in range(Nshot):
                                zcon.append(zconstant[:, r1, r2, i])
                                ztime.append(ztimeseries[:, r1, r2, i])
                            return zcon, ztime
                        zconstant, ztimeseries = shotcompile()
                        print(np.shape(zconstant))
                        tmod = []
                        zreal = []
                        # This creates so we can look at certain regions and then do analysis on each, Normally you could just render with limits, but since we analysis we need only specific region data
                        for j in range(Nshot):
                            for i in range(len(t)):
                                if(t[i] >= float(heldmin) and t[i] <= float(heldmax)):
                                    zreal.append(
                                        (ztimeseries[j][i])/zconstant[j][i])
                            skews = skew(zreal)
                            kurts = kurtosis(zreal, fisher=True)
                            skewkurtarray.append([skews, kurts, a])
                for b in range(len(skewkurtarray)):
                    if(skewkurtarray[b][2] == 0):
                        skew0x.append(skewkurtarray[b][0])
                        kurt0x.append(skewkurtarray[b][1])
                    if(skewkurtarray[b][2] == 1):
                        skew1x.append(skewkurtarray[b][0])
                        kurt1x.append(skewkurtarray[b][1])
            else:
                pass
            c4 = fig.add_subplot(gs[0, 3])
            plt.plot(skew0x, kurt0x, 'o')
            p = np.polyfit(skew0x, kurt0x, 2)
            skewline0 = np.linspace(
                np.min(skew0x), np.max(skew0x), len(kurt0x))
            yfit0 = np.polyval(p, skewline0)
            plt.plot(skewline0, yfit0, label=('a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                              str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3))))
            plt.legend()
            #plt.errorbar(skew0x, kurt0x,res, fmt='none')
            c5 = fig.add_subplot(gs[1, 3])
            plt.plot(skew1x, kurt1x, 'o')
            p = np.polyfit(skew1x, kurt1x, 2)
            skewline1 = np.linspace(
                np.min(skew1x), np.max(skew1x), len(kurt1x))
            yfit1 = np.polyval(p, skewline1)
            plt.plot(skewline1, yfit1, label=('a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                              str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3))))
            plt.legend()
            #plt.errorbar(skew1x, kurt1x, res, fmt='none')

            def GraphSetting():
                c4.set(xlabel="Skew")
                c4.set(ylabel="Kurtosis")

                c5.set(xlabel="Skew")
                c5.set(ylabel="Kurtosis")
            GraphSetting()

        def animations(i):
            plt.clf()
            x_choose = xmin+(xmax-xmin)*(i+1)/21
            CreateSkewKurt(i, skewkurtarray)
            CreatePlot(y_choose, x_choose)
        anim = animation.FuncAnimation(
            fig, animations, frames=len(y), repeat=False)
        anim.save('Animation.gif', writer=PillowWriter(fps=3))
        AnimationDisp(len(y))

    def AnimationTime():
        skewkurtarray = []

        def CreatePlotTSeries(index):
            def radius():
                ind1 = []
                r = []

                r_choose = np.sqrt(x_choose**2+y_choose**2)

                for i in range(len(x)):
                    for j in range(len(y)):
                        r.append(
                            [np.abs(np.sqrt(x[i]**2+y[j]**2)-float(r_choose)), i, j])  # gives all radi and their postions in the x y arrays

                for i in range(len(r)):
                    ind1.append((r[i][0]))  # appends the radii alone
                ind = np.argmin(ind1)  # fins the smallest radii
                # finds corresponding i and j values
                r1, r2 = r[ind][1], r[ind][2]
                return r1, r2
            r1, r2 = radius()
            skewkurtarray = []

            for a in range(len(nplots)):
                dataconstant = dataarray[a][1]
                data = dataarray[a][0]

                ztimeseries = np.reshape(
                    data, (Nt, Nx, Ny, Nshot), order="F")

                zconstant = np.reshape(
                    dataconstant, (Nt, Nx, Ny, Nshot), order="F")

                zheat = np.reshape(
                    dataconstant, (Nt, Nx, Ny, Nshot), order="F")  # for heatmap
                # find max, then set condition in
                heldinx = np.abs(t-heldchoose).argmin()
                zheat = zheat[heldinx, :, :, int(shot)-1]

                def shotcompliing():
                    zcon = []
                    ztime = []
                    for i in range(Nshot):
                        zcon.append(zconstant[:, r1, r2, i])
                        ztime.append(ztimeseries[:, r1, r2, i])
                    return zcon, ztime

                zconstant, ztimeseries = shotcompliing()

                print(np.shape(zconstant), np.shape(ztimeseries))
                tmod = []
                zreal = []
                zmod = []
                heldinx = np.abs(t-heldchoose).argmin()

                def c3def():
                    heldminmod = heldmin+(heldmax-5-heldmin)*(index+1)/21
                    heldmaxmod = heldminmod+5
                    for j in range(Nshot):
                        print(j)
                        # This creates so we can look at certain regions and then do analysis on each, Normally you could just render with limits, but since we analysis we need only specific region data
                        for i in range(len(t)):
                            if(t[i] >= float(heldminmod) and t[i] <= float(heldmaxmod)):
                                if j == int(shot)-1:
                                    tmod.append(t[i])
                                    zmod.append(
                                        (ztimeseries[j][i])/zconstant[j][i])
                                zreal.append(
                                    (ztimeseries[j][i])/zconstant[j][i])

                    c3 = fig.add_subplot(gs[a, 1])
                    plt.plot(tmod, zmod, color='black', label="First trial"+"@"+str(np.around(
                        np.sqrt(x[r1]**2+y[r2]**2), 4)))
                    plt.legend()
                    return c3, zreal

                def c2def(zreal):
                    c2 = fig.add_subplot(gs[a, 2])
                    sigma = np.std((zreal))
                    mu = np.mean((zreal))
                    # logmu = 10**(sigma+mu**2/2)
                    # logsigma = np.sqrt((10**(sigma**2)-1)*logmu*2)
                    zreal = zreal/sigma
                    # mean, var, skew, kurt = norm.stats(moments='mvsk')
                    x = np.linspace(np.min(zreal), np.max(zreal), 300)
                    plt.plot(x, norm.pdf(x), 'r--', lw=4, alpha=1)
                    count, bins, ignored = plt.hist(
                        zreal, density=True, bins=100, histtype='step', alpha=0.8, lw=2, color='black', label=("skew"+"="+str(np.around(skew(zreal), 4))+","+"kurtosis" +
                                                                                                               "="+str(np.around(kurtosis(zreal, fisher=True), 4))))
                    plt.legend()
                    count = count[count != 0]
                    skews = skew(zreal)
                    kurts = kurtosis(zreal, fisher=True)
                    skewkurtarray.append([skews, kurts, a])
                    return c2, count, skewkurtarray

                def c1def():
                    c1 = fig.add_subplot(gs[a, 0])
                    if(vector == True):
                        [xgrad, ygrad] = np.gradient(zheat)
                        plt.quiver(x, y, xgrad, ygrad, width=0.01)
                    else:
                        plt.pcolormesh(x, y, zheat, shading="gouraud",
                                       cmap='hot')  # heatmap
                        cbar = plt.colorbar()
                        cbar.ax.set_title(
                            str(nplots[a][1])+"($"+nplots[a][2]+"$)")

                # # Constants that show where we are on heatmap
                    plt.plot((x),
                             [y_choose]*len(y), color="black")
                    # # X/Y Slices, float array is simply making a line at the chosen value with the correct length(flat line on graph)
                    plt.plot([x_choose]*len(x), (y), color="black")
                    return c1

                def GraphSettings(c1, c2, c3, count):
                    c1.set(xlim=(xmin, xmax))
                    c1.set(ylim=(ymin, ymax))
                    c1.set(xlabel=xaxis+"("+xunit+")")
                    c1.set(ylabel=yaxis+"("+yunit+")")
                    c1.set(title="trial @"+shot+"and" +
                           topaxis+"("+heldunit+")")  # heatmap

                    c3.set(xlabel=topaxis+"("+heldunit+")")
                    c3.set(
                        ylabel=("$\\frac{\\delta"+" "+nplots[a][0]+"}{"+nplots[a][1]+"}"+"$"+"Arb. Units"))  # Time series
                    c3.set(title="snapshot of trial"+shot)

                    c2.set(
                        xlabel="$\\frac{\\delta"+" "+nplots[a][0]+"}{"+nplots[a][1]+"\\times \\sigma"+"}"+"$")  # PDF
                    c2.set(ylabel="log10[PDF]")
                    c2.set(title="lognormal PDF of all trials")
                    c2.set(yscale="log")
                    c2.set(ylim=(np.min(count), 1))
                c3, zreal = c3def()
                c2, count, skewkurtarray = c2def(zreal)
                c1 = c1def()
                GraphSettings(c1, c2, c3, count)
            return fig, skewkurtarray

        def CreateSkewKurt(index, skewkurtarray):
            if(index == 0):
                global skew0x, kurt0x, skew1x, kurt1x
                skew0x = []
                kurt0x = []
                skew1x = []
                kurt1x = []

                def radius():
                    ind1 = []
                    r = []

                    r_choose = np.sqrt(
                        x_choose**2+y_choose**2)

                    for j in range(len(x)):
                        for k in range(len(y)):
                            r.append(
                                [np.abs(np.sqrt(x[j]**2+y[k]**2)-float(r_choose)), j, k])  # gives all radi and their postions in the x y arrays

                    for j in range(len(r)):
                        ind1.append((r[j][0]))  # appends the radii alone

                    ind = np.argmin(ind1)  # fins the smallest radii
                    # finds corresponding i and j values
                    r1, r2 = r[ind][1], r[ind][2]
                    return r1, r2
                r1, r2 = radius()
                for a in range(len(nplots)):
                    data = frame[[nplots[a][0]]].to_numpy()
                    data = data[~np.isnan(data)]
                    dataconstant = frame[[nplots[a][1]]].to_numpy()
                    dataconstant = dataconstant[~np.isnan(dataconstant)]

                    ztimeseries = np.reshape(
                        data, (Nt, Nx, Ny, Nshot), order="F")

                    zconstant = np.reshape(
                        dataconstant, (Nt, Nx, Ny, Nshot), order="F")

                    def shotcompile():
                        zcon = []
                        ztime = []
                        for i in range(Nshot):
                            zcon.append(zconstant[:, r1, r2, i])
                            ztime.append(ztimeseries[:, r1, r2, i])
                        return zcon, ztime
                    zconstant, ztimeseries = shotcompile()

                    for j in range(21):
                        heldminmod = heldmin+(heldmax-5-heldmin)*(j+1)/21
                        heldmaxmod = heldminmod+5
                        tmod = []
                        zreal = []
                        for k in range(Nshot):
                            for i in range(len(t)):
                                if(t[i] >= float(heldminmod) and t[i] <= float(heldmaxmod)):
                                    tmod.append(t[i])
                                    zreal.append(
                                        (ztimeseries[k][i])/zconstant[k][i])
                            skews = skew(zreal)
                            kurts = kurtosis(zreal)
                            skewkurtarray.append([skews, kurts, a])
                for b in range(len(skewkurtarray)):
                    if(skewkurtarray[b][2] == 0):
                        skew0x.append(skewkurtarray[b][0])
                        kurt0x.append(skewkurtarray[b][1])
                    if(skewkurtarray[b][2] == 1):
                        skew1x.append(skewkurtarray[b][0])
                        kurt1x.append(skewkurtarray[b][1])
            else:
                pass
            c4 = fig.add_subplot(gs[0, 3])
            plt.plot(skew0x, kurt0x, 'o')
            p = np.polyfit(skew0x, kurt0x, 2)
            skewline0 = np.linspace(
                np.min(skew0x), np.max(skew0x), len(kurt0x))
            yfit0 = np.polyval(p, skewline0)
            plt.plot(skewline0, yfit0, label=('a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                              str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3))))
            plt.legend()
            #plt.errorbar(skew0x, kurt0x,res, fmt='none')
            c5 = fig.add_subplot(gs[1, 3])
            plt.plot(skew1x, kurt1x, 'o')
            p = np.polyfit(skew1x, kurt1x, 2)
            skewline1 = np.linspace(
                np.min(skew1x), np.max(skew1x), len(kurt1x))
            yfit1 = np.polyval(p, skewline1)
            plt.plot(skewline1, yfit1, label=('a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                              str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3))))
            plt.legend()
            #plt.errorbar(skew1x, kurt1x, res, fmt='none')

            def GraphSetting():
                c4.set(xlabel="Skew")
                c4.set(ylabel="Kurtosis")

                c5.set(xlabel="Skew")
                c5.set(ylabel="Kurtosis")
            GraphSetting()

        def animations(i):
            plt.clf()
            CreateSkewKurt(i, skewkurtarray)
            CreatePlotTSeries(i)
        anim = animation.FuncAnimation(
            fig, animations, frames=21, repeat=False)
        anim.save('Animation.gif', writer=PillowWriter(fps=3))
        AnimationDisp(len(y))

    def Plot():
        canvas1 = FigureCanvasTkAgg(fig1, master=timeheat)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=2, column=1)
        Tk.update(timeheat)  # For xchange length logic
        Label(timeheat, text="Change the held Variable" +
              topaxis).grid(row=0, column=1)
        holdchange = Scale(timeheat, orient=HORIZONTAL, resolution=deltat, from_=heldmin, to=heldmax, length=int(
            canvas1.get_tk_widget().winfo_width()))  # Makes slider size of graph
        holdchange.set(heldchoose)
        holdchange.grid(row=1, column=1)

        Label(timeheat, text="Vertical Label".replace(" ", " \n"),
              wraplength=1).grid(row=2, column=3)  # x=C
        xchange = Scale(timeheat, resolution=deltat, orient=VERTICAL, length=int(canvas1.get_tk_widget().winfo_height(
        )), from_=xmin, to=xmax)  # Because vertical line we use ymin and max
        xchange.set(x_choose)
        xchange.grid(row=2, column=2)
        Tk.update(timeheat)
        Label(timeheat, text="Change the Horizontal plot").grid(
            row=4, column=1)  # y=C
        ychange = Scale(timeheat, resolution=deltat, orient=HORIZONTAL, length=int(canvas1.get_tk_widget().winfo_width(
        )), from_=ymin, to=ymax)  # Because vertical line we use ymin and max
        ychange.set(y_choose)
        ychange.grid(row=3, column=1)
        Tk.update(timeheat)

        Label(timeheat, text="xmin").grid(row=0, column=5)
        xminchange = Entry(timeheat, width=5)
        xminchange.insert(END, xmin)
        xminchange.grid(row=0, column=6)
        Tk.update(timeheat)
        Label(timeheat, text="xmax").grid(row=1, column=5)
        xmaxchange = Entry(timeheat, width=5)
        xmaxchange.insert(END, xmax)
        xmaxchange.grid(row=1, column=6)
        Tk.update(timeheat)

        Label(timeheat, text="ymin").grid(row=0, column=7)
        yminchange = Entry(timeheat, width=5)
        yminchange.insert(END, ymin)
        yminchange.grid(row=0, column=8)
        Tk.update(timeheat)
        Label(timeheat, text="max").grid(row=1, column=7)
        ymaxchange = xmin
        ymaxchange = Entry(timeheat, width=5)
        ymaxchange.insert(END, ymax)
        ymaxchange.grid(row=1, column=8)
        Tk.update(timeheat)

        Label(timeheat, text="heldmin").grid(row=0, column=9)
        heldminchange = Entry(timeheat, width=5)
        heldminchange.insert(END, heldmin)
        heldminchange.grid(row=0, column=10)
        Tk.update(timeheat)
        Label(timeheat, text="heldmax").grid(row=1, column=9)
        heldmaxchange = xmin
        heldmaxchange = Entry(timeheat, width=5)
        heldmaxchange.insert(END, heldmax)
        heldmaxchange.grid(row=1, column=10)
        Tk.update(timeheat)

        Button(timeheat, text="animation X=x_chosen loop through y's ",
               command=lambda: AnimationX()).grid(row=4, column=1)
        Button(timeheat, text="animation Y=ychosen, loops through x",
               command=lambda: AnimationY()).grid(row=5, column=1)
        Button(timeheat, text="Loops through time 21, 5(unit) windows ",
               command=lambda: AnimationTime()).grid(row=6, column=1)
        changedisplacement = BooleanVar(timeheat)
        Checkbutton(timeheat, text='crosshair becomes origin if true',
                    variable=changedisplacement).grid(row=3, column=5)

        Button(timeheat, text="Re-render graphs",
               command=lambda: [ClearGraph(canvas1),
                                HeattimeGrapher(xaxis, yaxis, topaxis, xminchange.get(), yminchange.get(), xmaxchange.get(), ymaxchange.get(), deltat, float(xchange.get()),
                                                float(ychange.get()), float(holdchange.get()), heldmaxchange.get(
                                ), heldminchange.get(), datavar1, shot, vector, xunit, xfac, yunit, yfac,
                   heldunit, heldfac, False, constantvar1, datavar2, constantvar2, x, y, t, dataarray, varunit1, varunit1fac, varunit2, varunit2fac, changedisplacement.get())]).grid(row=8, column=1)

        quit = Button(timeheat, text="Quit", command=timeheat.destroy)
        quit.grid(row=9, column=1)
        Label(timeheat, text="Save this Figure").grid(row=2, column=4)
        Button(timeheat, command=lambda: plt.savefig("heatmap vs timeseries"+"@" +
                                                     xaxis+"="+str(x_choose)+","+str(yaxis)+"="+str(y_choose)+".png")).grid(row=2, column=5)
    Plot()


def TimeseriesSetting():
    # convert pandas dataframe to simple python list
    column_list = frame.columns.tolist()

    OPTIONS = column_list  # this is what solved my problem
    Label(timesetting, text="What radius would you like?").pack()
    radius = Entry(timesetting, width=10)
    radius.pack()

    Label(timesetting, text="Constant 1(should form radius)").pack()
    xaxis = StringVar(timesetting)
    OptionMenu(timesetting, xaxis, *OPTIONS).pack()

    Label(timesetting, text="Constant 1(should form radius)").pack()
    yaxis = StringVar(timesetting)
    OptionMenu(timesetting, yaxis, *OPTIONS).pack()

    Label(timesetting, text="X-axis variable").pack()
    held = StringVar(timesetting)
    OptionMenu(timesetting, held, *OPTIONS).pack()

    Label(timesetting, text="Y axis variable").pack()
    data = StringVar(timesetting)
    OptionMenu(timesetting, data, *OPTIONS).pack()

    Label(timesetting, text="Y axis Constant(should be DC)").pack()
    constantvar = StringVar(timesetting)
    OptionMenu(timesetting, constantvar, *OPTIONS).pack()

    Label(timesetting, text="Xmin").pack()
    xmin = Entry(timesetting, width=10)
    xmin.pack()
    Label(timesetting, text="Xmax").pack()
    xmax = Entry(timesetting, width=10)
    xmax.pack()

    Label(timesetting, text="Trial number").pack()
    trial = Entry(timesetting, width=10)
    trial.pack()

    Button(timesetting, text="Quit", command=lambda: timesetting.destroy).pack()
    Button(timesetting, text="Commence", command=lambda: [skewCreator(), SkewKurt(
        radius.get(), xaxis.get(), yaxis.get(), held.get(), data.get(), trial.get(), xmin.get(), xmax.get(), constantvar.get())
    ]).pack()

    for i in range(len(OPTIONS)):
        maxmin = Label(timesetting, text=(OPTIONS[i]+" "+"Has a max of"+" "+str(float(frame[OPTIONS[i]].max())) +
                                          " "+"and a min of"+" "+str(frame[OPTIONS[i]].min())))
        maxmin.pack()


def skewCreator():
    global skewkurt
    skewkurt = Toplevel()
    skewkurt.title("Heat map")
    skewkurt.geometry('1500x1000')
    return skewkurt


def SkewKurt(r_chosen, xaxis, yaxis, topaxis, datavar, shot, xmin, xmax, constantvar):
    def Graphs():
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, hspace=0.99, wspace=0.5, bottom=0.15)  # Formatting

        x = frame[[xaxis]].to_numpy()
        x = x[~np.isnan(x)]
        y = frame[[yaxis]].to_numpy()
        y = y[~np.isnan(y)]
        t = frame[[topaxis]].to_numpy()
        t = t[~np.isnan(t)]
        data = frame[[datavar]].to_numpy()
        data = data[~np.isnan(data)]
        dataconstant = frame[[constantvar]].to_numpy()
        dataconstant = dataconstant[~np.isnan(dataconstant)]

        Nt = len(t)
        Nx = len(x)
        Ny = len(y)
        z = np.reshape(data, (Nt, Nx, Ny, Nshot), order="F")
        zconstant = np.reshape(dataconstant, (Nt, Nx, Ny, Nshot))
        ind1 = []

        def radius():
            r = []
            for i in range(len(x)):
                for j in range(len(y)):
                    r.append(
                        [np.abs(np.sqrt(x[i]**2+y[j]**2)-float(r_chosen)), i, j])

            for i in range(len(r)):
                ind1.append((r[i][0]))
            ind = np.argmin(ind1)
            r1, r2 = r[ind][1], r[ind][2]  # Gives radius
            return r1, r2
        r1, r2 = radius()

        # SELECTS VALUES From matrix to give 1D array
        z = z[:, r1, r2, int(shot)-1]
        # Constant for Operations
        zconstant = zconstant[:, r1, r2, int(shot)-1]

        def TimeSeries():
            tmod = []
            zreal = []
            for i in range(len(t)):  # This creates so we can look at certain regions and then do analysis on each, Normally you could just render with limits, but since we analysis we need only specific region data
                if(t[i] >= float(xmin) and t[i] <= float(xmax)):
                    tmod.append(t[i])
                    zreal.append((z[i])/zconstant[i])  # OPEATION

            c1 = fig.add_subplot(gs[0, 0])
            radiusclosest = np.sqrt(x[r1]**2+y[r2]**2)
            plt.plot(tmod, zreal, color='black',
                     label=np.around(radiusclosest, 4))
            plt.legend()
            return c1, zreal

        def PDF(zreal):
            c2 = fig.add_subplot(gs[0, 1])
            sigma = np.std((zreal))
            mu = np.mean((zreal))
            #logmu = 10**(sigma+mu**2/2)
            #logsigma = np.sqrt((10**(sigma**2)-1)*logmu*2)
            zreal = zreal/sigma
            #mean, var, skew, kurt = norm.stats(moments='mvsk')
            x = np.linspace(np.min(zreal), np.max(zreal), 300)
            plt.plot(x, norm.pdf(x), 'r--', lw=4, alpha=1)
            count, bins, ignored = plt.hist(
                zreal, density=True, bins=100, histtype='step', alpha=0.8, lw=2, color='black', label=("skew"+"="+str(np.around(skew(zreal), 4))+" "+"and"+" "+"kurtosis" +
                                                                                                       "="+str(np.around(kurtosis(zreal, fisher=True), 4))))
            plt.legend()
            count = np.asarray(count)
            count = count[count != 0]
            return c2, count

        def GraphSetting(c1, c2, count):
            c1.set(xlabel=topaxis)
            c1.set(ylabel="Probability of"+" " +
                   datavar+" "+"and"+" "+constantvar)
            c2.set(xlabel="Normalized Amplitude")
            c2.set(ylabel="log10[PDF]")
            c2.set(title="Amplitude/sigma probability")
            c2.set(yscale="log")
            c2.set(ylim=(count.min(), 1))
        c1, zreal = TimeSeries()
        c2 = PDF(zreal)
        c1, zreal = TimeSeries()
        c2, count = PDF(zreal)
        GraphSetting(c1, c2, count)
        return fig

    def Plot():
        fig = Graphs()
        canvas1 = FigureCanvasTkAgg(fig, master=skewkurt)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=2, column=1)
        Label(skewkurt, text="Save this Figure").grid(row=2, column=4)
        Button(skewkurt, command=lambda: plt.savefig("Time series of" +
                                                     datavar+"and a radius of"+r_chosen+".png")).grid(row=2, column=5)

        Label(skewkurt, text="xmin").grid(row=0, column=5)
        xminchange = Entry(skewkurt, width=5)
        xminchange.insert(END, xmin)
        xminchange.grid(row=1, column=5)
        Tk.update(skewkurt)

        Label(skewkurt, text="xmax").grid(row=0, column=6)
        xmaxchange = Entry(skewkurt, width=5)
        xmaxchange.insert(END, xmax)
        xmaxchange.grid(row=1, column=6)
        Tk.update(skewkurt)

        Label(skewkurt, text="radius").grid(row=0, column=7)
        radiuschange = Entry(skewkurt, width=5)
        radiuschange.insert(END, r_chosen)
        radiuschange.grid(row=1, column=7)
        Tk.update(skewkurt)

        Button(skewkurt, text="Quit",
               command=skewkurt.destroy).grid(row=0, column=10)
        Button(skewkurt, text="re-render graphs", command=lambda: [ClearGraph(canvas1), SkewKurt(
            radiuschange.get(), xaxis, yaxis, topaxis, datavar, shot, xminchange.get(), xmaxchange.get(), constantvar)
        ]).grid(row=1, column=10)
    Plot()


def Go_button(graph):
    if(graph == "2d heatmap"):
        global heatwindow
        heatwindow = Toplevel()
        heatwindow.title("heat map settings")
        heatwindow.geometry('1920x1080')
        Heatsetting(heatwindow)
    if(graph == "Skew/Kurt"):
        global timesetting
        timesetting = Toplevel()
        timesetting.title("Kurtosis and Skew Settings")
        timesetting.geometry('1920x1080')
        print("ya")
        TimeseriesSetting()
    if(graph == "heat vs time"):
        global heatimesetting
        heatimesetting = Toplevel()
        heatimesetting.title("heatmap and time series Settings")
        heatimesetting.geometry('1920x1080')
        print("ya")
        heatTimeSetting()
    if(graph == "Mode animation"):
        global modeheats
        modeheats = Toplevel()
        modeheats.title("heatmap mode Settings")
        modeheats.geometry('1920x1080')
        print("ya")
        Headmodesetting()
    if(graph == 'radiusgraphs'):
        global radiussetting
        radiussetting = Toplevel()
        radiussetting.title("radius settings")
        radiussetting.geometry('1920x1080')
        radiussettings()


def Graph_select():

    plot_variable = StringVar(root)
    plot_variable.set("")
    plot_type = OptionMenu(root, plot_variable,
                           "2d heatmap", "Skew/Kurt", "heat vs time", "Mode animation", 'radiusgraphs').pack()

    Button(root, text="Commence", command=lambda: [
           Go_button(plot_variable.get())]).pack()
    Button(root, text="Go back to File loader",
           command=lambda: FileLoader()).pack()


def Headmodesetting():
    column_list = frame.columns.tolist()

    OPTIONS = column_list  # this is what solved my problem
    x_axis_var = StringVar(modeheats)
    Label(modeheats, text="Variable of x-axis").grid(row=0, column=0)
    OptionMenu(modeheats, x_axis_var,
               *OPTIONS).grid(row=1, column=0)

    Label(modeheats, text="x-axis min").grid(row=3, column=0)
    xmin = Entry(modeheats, width=5)
    xmin.grid(row=4, column=0)

    Label(modeheats, text="x-axis max").grid(row=5, column=0)
    xmax = Entry(modeheats, width=5)
    xmax.grid(row=6, column=0)

    Label(modeheats, text="x unit").grid(row=7, column=0)
    xunit = Entry(modeheats, width=5)
    xunit.grid(row=8, column=0)

    Label(modeheats, text="x unit factor(ex: cm->m= 100)").grid(row=9, column=0)
    xfac = Entry(modeheats, width=5)
    xfac.grid(row=10, column=0)

    y_axis_var = StringVar(modeheats)
    Label(modeheats, text="Variable of heatmap y-axis").grid(row=0, column=1)
    OptionMenu(modeheats, y_axis_var,
               *OPTIONS).grid(row=1, column=1)

    Label(modeheats, text=" heatmap ymin").grid(row=3, column=1)
    ymin = Entry(modeheats, width=5)
    ymin.grid(row=4, column=1)

    Label(modeheats, text="ymax").grid(row=5, column=1)
    ymax = Entry(modeheats, width=5)
    ymax.grid(row=6, column=1)

    Label(modeheats, text="y unit").grid(row=7, column=1)
    yunit = Entry(modeheats, width=5)
    yunit.grid(row=8, column=1)

    Label(modeheats, text="y unit factor(ex: cm->m= 100)").grid(row=9, column=1)
    yfac = Entry(modeheats, width=5)
    yfac.grid(row=10, column=1)

    Label(modeheats, text="Held Variable").grid(
        row=0, column=2)  # Thing we will be scrolling through
    held_var = StringVar(modeheats)
    OptionMenu(modeheats, held_var, *OPTIONS).grid(
        row=1, column=2)

    Label(modeheats, text="Held Variable Min").grid(
        row=4, column=2)  # For scroll function
    held_choosemin = Entry(modeheats, width=5)
    held_choosemin.grid(row=5, column=2)

    Label(modeheats, text="Held Variable Max").grid(
        row=6, column=2)  # For scroll function
    held_choosemax = Entry(modeheats, width=5)
    held_choosemax.grid(row=7, column=2)

    Label(modeheats, text="held unit").grid(row=8, column=2)
    heldunit = Entry(modeheats, width=5)
    heldunit.grid(row=9, column=2)

    Label(modeheats, text="held unit factor(ex: cm->m= 100)").grid(row=10, column=2)
    heldfac = Entry(modeheats, width=5)
    heldfac.grid(row=11, column=2)
    try:
        ind = OPTIONS.index('x')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('X')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('Y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('Time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('T')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('t')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    if(parmst == True):
        held_choosemin.insert(END, tminn)
        held_choosemax.insert(END, tmaxn)
    heldfac.insert(END, 1)
    if(parmsx == True):
        xmin.insert(END, xminn)
        xmax.insert(END, xmaxn)
    xfac.insert(END, 1)
    if(parmsy == True):
        ymin.insert(END, yminn)
        ymax.insert(END, ymaxn)
        yfac.insert(END, 1)

    Label(modeheats, text="data variable 1").grid(row=0, column=3)
    data_var1 = StringVar(modeheats)
    OptionMenu(modeheats, data_var1, *OPTIONS).grid(
        row=1, column=3)
    Label(modeheats, text="data variable 2").grid(row=2, column=3)
    data_var2 = StringVar(modeheats)
    OptionMenu(modeheats, data_var2, *OPTIONS).grid(
        row=3, column=3)
    Label(modeheats, text="data variable 3").grid(row=4, column=3)
    data_var3 = StringVar(modeheats)
    OptionMenu(modeheats, data_var3, *OPTIONS).grid(
        row=5, column=3)
    Label(modeheats, text="data variable 4").grid(row=6, column=3)
    data_var4 = StringVar(modeheats)
    OptionMenu(modeheats, data_var4, *OPTIONS).grid(
        row=7, column=3)

    Label(modeheats, text="data unit").grid(row=8, column=3)
    dataunit = Entry(modeheats, width=15)
    dataunit.grid(row=9, column=3)
    Label(modeheats, text="data unit factor(ex: cm->m= 100)").grid(row=10, column=3)
    datafac = Entry(modeheats, width=5)
    datafac.grid(row=11, column=3)

    Label(modeheats, text="If multiple trials, which trial").grid(
        row=0, column=6)  # For scroll function
    trial = Entry(modeheats, width=5)
    trial.grid(row=1, column=6)

    for i in range(len(OPTIONS)):
        maxmin = Label(modeheats, text=(OPTIONS[i]+" "+"Has a max of"+" "+str(float(frame[OPTIONS[i]].max())) +
                                        " "+"and a min of"+" "+str(frame[OPTIONS[i]].min())))
        maxmin.grid(row=i, column=4)

    vector = BooleanVar(modeheats)
    Checkbutton(
        modeheats, text="Would you like enable a vector map", variable=vector).grid(row=4, column=6)
    vectorimpose = BooleanVar(modeheats)
    Checkbutton(modeheats, text="Would you like to superimpose a vector map",
                variable=vectorimpose).grid(row=5, column=6)
    maxmin = BooleanVar(modeheats)
    Checkbutton(modeheats, text="Would you like to set the animation to have a set color-bar",
                variable=maxmin).grid(row=6, column=6)

    Button(modeheats, text="Start animation", command=lambda: heatmode(x_axis_var.get(), xmin.get(), xmax.get(),
                                                                       xunit.get(), xfac.get(), y_axis_var.get(
    ), ymin.get(), ymax.get(), yunit.get(), yfac.get(),
        held_var.get(), held_choosemin.get(), held_choosemax.get(), heldunit.get(), heldfac.get(), data_var1.get(
    ), data_var2.get(), data_var3.get(), data_var4.get(), dataunit.get(), datafac.get(), trial.get(),
        vector.get(), vectorimpose.get(), maxmin.get())).grid(row=10, column=10)


def heatmode(xaxis, xmin, xmax, xunit, xfac, yaxis, ymin, ymax, yunit, yfac, topaxis, heldmin, heldmax,
             heldunit, heldfac, var1, var2, var3, var4, datavar, datafac, shot, vector, vectorimpose, animationmaxmin):
    xmin, ymin, xmax, ymax, heldmax, heldmin = float(xmin), float(
        ymin), float(xmax), float(ymax), float(heldmax), float(heldmin)
    nplots = [[var1, var2], [var3, var4]]
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 2, wspace=0.2, hspace=0.5)
    x, y, t, dataarray = [], [], [], []
    heldchoose = 0
    Nx, Ny, Nt = 0, 0, 0
    vmin, vmax = 0, 0

    def constantfactor():
        nonlocal xmin, xmax, ymin, ymax, heldmax, heldmin
        xmin = xmin*float(xfac)
        xmax = xmax*float(xfac)
        ymin = ymin*float(yfac)
        ymax = ymax*float(yfac)
        heldmax = heldmax*float(heldfac)
        heldmin = heldmin*float(heldfac)

    def Varassigning():
        try:
            nonlocal x, y, t, dataarray, Nx, Ny, Nt
            x = frame[[xaxis]].to_numpy()
            x = x[~np.isnan(x)]
            y = frame[[yaxis]].to_numpy()
            y = y[~np.isnan(y)]
            t = frame[[topaxis]].to_numpy()
            t = t[~np.isnan(t)]

            x = x*float(xfac)
            y = y*float(yfac)
            t = t*float(heldfac)
            Nx, Ny, Nt = len(x), len(y), len(t)
            for a in range(len(nplots)):
                data1 = frame[[nplots[a][0]]].to_numpy()
                data1 = data1[~np.isnan(data1)]
                data2 = frame[[nplots[a][1]]].to_numpy()
                data2 = data2[~np.isnan(
                    data2)]
                data1 = data1*float(datafac)
                data2 = data2*float(datafac)
                dataarray.append([data1, data2])
            dataarray = np.asarray(dataarray)

        except KeyError:
            print("error")

    def create_plot():
        for i in range(len(dataarray)):
            for j in range(len(dataarray[i])):
                data = dataconditioning(i, j)
                c1 = plotGraph(data, i, j)
                plotsetting(c1)

    def dataconditioning(i, j):
        nonlocal vmin, vmax
        data = dataarray[i][j]
        if(animationmaxmin == True):
            vmin, vmax = np.min(data), np.max(data)
        data = np.reshape(data, (Nt, Nx, Ny, Nshot), order="F")
        heldinx = np.abs(t-heldchoose).argmin()
        data = data[heldinx, :, :, int(shot)-1]
        return data

    def plotGraph(data, i, j):
        c1 = fig.add_subplot(gs[i, j])
        if(vectorimpose == True):
            heatmap(data, i, j)
            quiver(data)
            print("test")

        elif(vector == True):
            quiver(data)

        else:
            heatmap(data, i, j)
        return c1

    def heatmap(data, i, j):
        if(animationmaxmin == True):
            plt.pcolormesh(x, y, data, shading="gouraud",
                           cmap='hot', vmin=vmin, vmax=vmax)  # heatmap
        else:
            plt.pcolormesh(x, y, data, shading="gouraud",
                           cmap='hot')
        cbar = plt.colorbar()
        cbar.ax.set_title(nplots[i][j]+"("+datavar+")")
        print("bam")

    def quiver(data):
        [xgrad, ygrad] = np.gradient(data)  # only do on DC plots nplots[a][1]
        plt.quiver(x, y, xgrad, ygrad, width=0.01)

    def plotsetting(c1):
        c1.set(xlabel=xaxis+"("+xunit+")")
        c1.set(title=topaxis+"@" +
               str(np.around(heldchoose, 4))+""+"("+heldunit+")")
        c1.set(ylabel=yaxis+"("+yunit+")")
        c1.set(xlim=(xmin, xmax))
        c1.set(ylim=(ymin, ymax))
        # constantfactor()
    constantfactor()
    Varassigning()

    def Animation():
        def animate(i):
            plt.clf()
            nonlocal heldchoose
            heldchoose = float(heldmin)+(float(heldmax) -
                                         float(heldmin))*(i+1)/21
            create_plot()
        anim = animation.FuncAnimation(
            fig, animate, frames=21, repeat=False)
        # saving function, first we render to screen
        anim.save('Animation.gif', writer=PillowWriter(fps=3))
        AnimationDisp(21)
    Animation()


def Heatsetting(heatwindow):
    # convert pandas dataframe to simple python list
    column_list = frame.columns.tolist()

    OPTIONS = column_list  # this is what solved my problem
    x_axis_var = StringVar(heatwindow)
    Label(heatwindow, text="Variable of x-axis").grid(row=0, column=0)
    OptionMenu(heatwindow, x_axis_var,
               *OPTIONS).grid(row=1, column=0)

    Label(heatwindow, text="Cross-sectional number for x-axis (must be between x min and max)").grid(row=2, column=0)
    x_choose = Entry(heatwindow, width=5)
    x_choose.grid(row=3, column=0)

    Label(heatwindow, text="x-axis min").grid(row=4, column=0)
    xmin = Entry(heatwindow, width=5)
    xmin.grid(row=5, column=0)

    Label(heatwindow, text="x-axis max").grid(row=7, column=0)
    xmax = Entry(heatwindow, width=5)
    xmax.grid(row=8, column=0)

    Label(heatwindow, text="x unit").grid(row=9, column=0)
    xunit = Entry(heatwindow, width=5)
    xunit.grid(row=10, column=0)

    Label(heatwindow, text="x unit factor(ex: cm->m= 100)").grid(row=11, column=0)
    xfac = Entry(heatwindow, width=5)
    xfac.grid(row=12, column=0)

    y_axis_var = StringVar(heatwindow)
    Label(heatwindow, text="Variable of heatmap y-axis").grid(row=0, column=1)
    OptionMenu(heatwindow, y_axis_var,
               *OPTIONS).grid(row=1, column=1)

    Label(heatwindow, text="Cross-sectional number for heatmap y-axis").grid(row=2, column=1)
    y_choose = Entry(heatwindow, width=5)
    y_choose.grid(row=3, column=1)

    Label(heatwindow, text=" heatmap ymin").grid(row=4, column=1)
    ymin = Entry(heatwindow, width=5)
    ymin.grid(row=6, column=1)

    Label(heatwindow, text="ymax").grid(row=7, column=1)
    ymax = Entry(heatwindow, width=5)
    ymax.grid(row=8, column=1)

    Label(heatwindow, text="y unit").grid(row=9, column=1)
    yunit = Entry(heatwindow, width=5)
    yunit.grid(row=10, column=1)

    Label(heatwindow, text="y unit factor(ex: cm->m= 100)").grid(row=11, column=1)
    yfac = Entry(heatwindow, width=5)
    yfac.grid(row=12, column=1)

    Label(heatwindow, text="Held Variable").grid(
        row=0, column=2)  # Thing we will be scrolling through
    held_var = StringVar(heatwindow)
    OptionMenu(heatwindow, held_var, *OPTIONS).grid(
        row=1, column=2)

    Label(heatwindow, text="Held Variable Value").grid(
        row=2, column=2)  # For scroll function
    held_choose = Entry(heatwindow, width=5)
    held_choose.grid(row=3, column=2)

    Label(heatwindow, text="Held Variable Min").grid(
        row=4, column=2)  # For scroll function
    held_choosemin = Entry(heatwindow, width=5)
    held_choosemin.grid(row=5, column=2)

    Label(heatwindow, text="Held Variable Max").grid(
        row=6, column=2)  # For scroll function
    held_choosemax = Entry(heatwindow, width=5)
    held_choosemax.grid(row=7, column=2)

    Label(heatwindow, text="held unit").grid(row=8, column=2)
    heldunit = Entry(heatwindow, width=5)
    heldunit.grid(row=9, column=2)

    Label(heatwindow, text="held unit factor(ex: cm->m= 100)").grid(row=10, column=2)
    heldfac = Entry(heatwindow, width=5)
    heldfac.grid(row=11, column=2)

    Label(heatwindow, text="Data Variable").grid(
        row=0, column=5)  # For scroll function
    datavar = StringVar(heatwindow)
    OptionMenu(heatwindow, datavar, *OPTIONS).grid(row=1, column=5)

    Label(heatwindow, text="If multiple trials, which trial").grid(
        row=2, column=5)  # For scroll function
    trial = Entry(heatwindow, width=5)
    trial.grid(row=3, column=5)

    Label(heatwindow, text="Resolution").grid(
        row=4, column=5)
    deltat = Entry(heatwindow, width=5)
    deltat.grid(row=5, column=5)
    vector = BooleanVar()
    Checkbutton(
        heatwindow, text="Would you like swap heatmap for a vector map", variable=vector).grid(row=6, column=5)
    vectorimpose = BooleanVar()
    Checkbutton(
        heatwindow, text="Would you like superimprove a vector map", variable=vectorimpose).grid(row=7, column=5)

    Label(heatwindow, text="Data Variable Max (for animation)").grid(
        row=8, column=5)  # For scroll function
    datamax = Entry(heatwindow, width=5)
    datamax.grid(row=9, column=5)

    Label(heatwindow, text="Data Variable min(for animation)").grid(
        row=10, column=5)  # For scroll function
    datamin = Entry(heatwindow, width=5)
    datamin.grid(row=11, column=5)
    try:
        ind = OPTIONS.index('x')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('X')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('Y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('Time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    try:
        ind = OPTIONS.index('T')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass
    try:
        ind = OPTIONS.index('t')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True

    except ValueError:
        pass

    if(parmst == True):
        held_choosemin.insert(END, tminn)
        held_choosemax.insert(END, tmaxn)
    heldfac.insert(END, 1)
    if(parmsx == True):
        xmin.insert(END, xminn)
        xmax.insert(END, xmaxn)
    xfac.insert(END, 1)
    if(parmsy == True):
        ymin.insert(END, yminn)
        ymax.insert(END, ymaxn)
        yfac.insert(END, 1)
    for i in range(len(OPTIONS)):
        maxmin = Label(heatwindow, text=(OPTIONS[i]+" "+"Has a max of"+" "+str(float(frame[OPTIONS[i]].max())) +
                                         " "+"and a min of"+" "+str(frame[OPTIONS[i]].min())))
        maxmin.grid(row=i, column=4)

    Button(heatwindow, text="Graph", command=lambda: [
        GraphHeatCreator(),
        Graph_heatmap(x_axis_var.get(), y_axis_var.get(), held_var.get(), xmin.get(), ymin.get(), xmax.get(), ymax.get(), deltat.get(),
                      x_choose.get(), y_choose.get(), held_choose.get(), held_choosemax.get(),
                      held_choosemin.get(), datavar.get(), trial.get(), vector.get(), xunit.get(), xfac.get(), yunit.get(), yfac.get(), heldunit.get(), heldfac.get(), True, datamin.get(), datamax.get(), vectorimpose.get()),  # False is just making sure scaling isnt exponential
        ErrorHeat(xmin.get(), ymin.get(), xmax.get(), ymax.get(), x_choose.get(), y_choose.get(), held_choose.get(), held_choosemax.get(), held_choosemin.get())]).grid(column=9, row=30)
    Button(graphheat, text="Quit", command=heatwindow.destroy).grid(
        column=9, row=31)


# Checks for any errors before anything happens
def ErrorHeat(xmin, ymin, xmax, ymax, xchoose, ychoose, heldchoose, heldmax, heldmin):
    print("For later")
#     if(
#         xmin <= xchoose and xchoose <= xmax or ymin <= ychoose and ychoose <= ymax
#     ):
#         pass
#     else:
#         graphheat.destroy
#         messagebox.showerror("Error", "Choose a (x,y) value in the bounds")
#     if(heldmin <= heldchoose and heldchoose <= heldmax):
#         pass
#     else:
#         graphheat.destroy
#         messagebox.showerror(
#             "Error", "Choose a held value in the bounds")  # bugged FIX


def GraphHeatCreator():
    global graphheat
    graphheat = Toplevel()
    graphheat.title("Heat map")
    graphheat.geometry('1500x1000')
    return graphheat


def Graph_heatmap(xaxis, yaxis, topaxis, xmin, ymin, xmax, ymax, deltat, x_choose, y_choose, heldchoose, heldmax, heldmin, datavar, shot, vector, xunit, xfac, yunit, yfac, heldunit, heldfac,
                  firstrender, datamin, datamax, vectorimpose):
    try:
        xmin, ymin, xmax, ymax, deltat, x_choose, y_choose, heldchoose, heldmax, heldmin, datamin, datamax = float(xmin), float(ymin), float(xmax), float(ymax), float(deltat), float(
            x_choose), float(y_choose), float(heldchoose), float(heldmax), float(heldmin), float(datamin), float(datamax)
        try:
            if(firstrender == True):
                xmin = xmin*float(xfac)
                xmax = xmax*float(xfac)
                ymin = ymin*float(yfac)
                ymax = ymax*float(yfac)
                heldmax = heldmax*float(heldfac)
                heldmin = heldmin*float(heldfac)
                y_choose = y_choose*float(yfac)
                x_choose = x_choose*float(xfac)
                heldchoose = heldchoose*float(heldfac)
        except ValueError:
            graphheat.destroy
            messagebox.showerror(
                message="Make sure all your factors are floats")

        def create_plot():  # n is number for animation
            # Slider for changing held variable
            gs = GridSpec(2, 2, hspace=0.99, wspace=0.3,
                          bottom=0.15)  # Formatting
            try:
                x = frame[[xaxis]].to_numpy()
                x = x[~np.isnan(x)]
                y = frame[[yaxis]].to_numpy()
                y = y[~np.isnan(y)]
                t = frame[[topaxis]].to_numpy()
                t = t[~np.isnan(t)]
                data = frame[[datavar]].to_numpy()
                data = data[~np.isnan(data)]
            except KeyError:
                graphheat.destroy
                messagebox.showerror(
                    message="Make sure every variable is assigned")
            x = x*float(xfac)
            y = y*float(yfac)
            t = t*float(heldfac)

            Nt = len(t)
            Nx = len(x)
            Ny = len(y)
            heldinx = np.abs(t-heldchoose).argmin()
            try:
                if(Nt > Nx and Nt > Ny):
                    z = np.reshape(data, (Nt, Nx, Ny, Nshot), order="F")
                elif(Nt >= Nx and Ny > Nt):
                    z = np.reshape(data, (Ny, Nt, Nx, Nshot))
                else:
                    z = np.reshape(data, (Nx, Nt, Ny, Nshot))
            except ValueError:
                graphheat.destroy
                messagebox.showerror(
                    message="Make sure that your data variable is the data variable, other variables can be arranged as pleased")
            try:
                z = z[heldinx, :, :, int(shot)-1]
            except ValueError:
                graphheat.destroy
                messagebox.showerror(message="Make sure shot is an integer")
            if(len(y) > len(x)):
                y = np.linspace(np.min(y), np.max(y), len(x))
                print("yup")
            else:
                x = np.linspace(np.min(x), np.max(x), len(y))
                print("nope")

            xdataind = (np.abs(x-x_choose)).argmin()
            xdata = z[:, xdataind]
            ydataind = (np.abs(y-y_choose)).argmin()
            ydata = z[ydataind, :]
            plt.clf()
            fig = plt.figure(figsize=(8, 4))

            c1 = fig.add_subplot(gs[0, 0])  # creates 1D slices
            plt.plot(y, xdata, color="black")

            c2 = fig.add_subplot(gs[1, 0])
            plt.plot(x, ydata, color="black")

            c3 = fig.add_subplot(gs[:, 1], aspect='equal')
            ax = plt.gca()
            ax.set_aspect(1)
            plt.axis('square')
            if(vectorimpose == True):
                plt.pcolormesh(x, y, z, shading="gouraud",
                               cmap='hot')  # heatmap
                plt.axis('equal')
                cbar = plt.colorbar()
                cbar.set_label(datavar, rotation=270)
                [xgrad, ygrad] = np.gradient(z)
                plt.quiver(x, y, xgrad, ygrad, width=0.01)
            elif(vector == True):
                [xgrad, ygrad] = np.gradient(z)
                plt.quiver(x, y, xgrad, ygrad, width=0.01)
            else:
                plt.pcolormesh(x, y, z, shading="gouraud",
                               cmap='hot')  # heatmap
                cbar = plt.colorbar()
                cbar.set_label(datavar, rotation=270)

            # # Constants that show where we are on heatmap
            plt.plot((x),
                     [y_choose]*len(y), color="black")
            # # X/Y Slices, float array is simply making a line at the chosen value with the correct length(flat line on graph)
            plt.plot([x_choose]*len(x), (y), color="black")

            def GraphSetting():
                c3.set(xlabel=xaxis+xunit)
                c3.set(title=topaxis+"@"+str(heldchoose)+""+"("+heldunit+")")
                c3.set(ylabel=yaxis+yunit)
                c3.set(xlim=(xmin, xmax))
                c3.set(ylim=(ymin, ymax))

                c1.set_xlabel(yaxis+" "+"("+yunit+")")

                c1.set(title=xaxis+""+"("+xunit+")"+"="+str(x_choose))
                c1.set(xlim=(ymin, ymax))
                c1.set(ylabel=datavar)

                c2.set(xlabel=xaxis+" "+"("+xunit+")"+"c1")

                c2.set(title=yaxis+" "+"("+yunit+")"+"="+str(y_choose))

                c2.set(xlim=(xmin, xmax))
                c2.set(ylabel=datavar)

            GraphSetting()

            return fig, z
    except ValueError:
        graphheat.destroy
        messagebox.showerror(message="Make sure every value is inputted")
    fig1, z = create_plot()

    def Animation(x_choose, y_choose, heldchoose, xmin, xmax, ymin, ymax):
        fig = plt.figure(figsize=(5, 3))
        gs = GridSpec(2, 2, hspace=0.99, wspace=0.3,
                      bottom=0.15)  # Formatting
        x = frame[[xaxis]].to_numpy()
        x = x[~np.isnan(x)]
        y = frame[[yaxis]].to_numpy()
        y = y[~np.isnan(y)]
        t = frame[[topaxis]].to_numpy()
        t = t[~np.isnan(t)]
        data = frame[[datavar]].to_numpy()
        data = data[~np.isnan(data)]
        global xani
        global yani
        global tani
        xani = x*float(xfac)
        yani = y*float(yfac)
        tani = t*float(heldfac)

        Nt = len(t)
        Nx = len(x)
        Ny = len(y)
        global vmin, vmax
        vmax, vmin = np.max(data), np.min(data)
        z = np.reshape(data, (Nt, Nx, Ny, Nshot), order="F")

        global dataani
        dataani = z  # because cant call props into animate

        def animateinter(i):
            plt.clf()
            global xani
            global yani
            global tani
            plt.clf()
            index = i*48
            z = dataani[index, :, :, int(shot)-1]
            if(Nt > Nx and Nt > Ny):
                z = np.reshape(data, (Nt, Nx, Ny, Nshot), order="F")
            elif(Nt >= Nx and Ny > Nt):
                z = np.reshape(data, (Ny, Nt, Nx, Nshot))
            else:
                z = np.reshape(data, (Nx, Nt, Ny, Nshot))

            if(len(yani) > len(xani)):
                yani = np.linspace(np.min(yani), np.max(yani), len(xani))
            else:
                xani = np.linspace(np.min(xani), np.max(xani), len(yani))

            z = z[index, :, :, int(shot)]
            xdataind = (np.abs(xani-x_choose)).argmin()
            xdata = z[:, xdataind]
            ydataind = (np.abs(yani-y_choose)).argmin()
            ydata = z[ydataind, :]

            c1 = fig.add_subplot(gs[0, 0])  # creates 1D slices
            plt.plot(yani, xdata, color="black")

            c2 = fig.add_subplot(gs[1, 0])
            plt.plot(xani, ydata, color="black")

            c3 = fig.add_subplot(gs[:, 1])

            # Quiver
            plt.axis('square')
            if(vectorimpose == True):
                plt.pcolormesh(x, y, z, shading="gouraud",
                               cmap='hot', vmin=datamin, vmax=datamax)  # heatmap
                cbar = plt.colorbar()
                cbar.set_label(datavar, rotation=270)
                [xgrad, ygrad] = np.gradient(z)
                plt.quiver(x, y, xgrad, ygrad, width=0.01)
            elif(vector == True):
                [xgrad, ygrad] = np.gradient(z)
                plt.quiver(x, y, xgrad, ygrad, width=0.01)
            else:
                plt.pcolormesh(x, y, z, shading="gouraud",
                               cmap='hot', vmin=datamin, vmax=datamax)  # heatmap
                cbar = plt.colorbar()
                cbar.set_label(datavar, rotation=270)

            # # Constants that show where we are on heatmap
            plt.plot((x),
                     [y_choose]*len(y), color="black")
            # # X/Y Slices, float array is simply making a line at the chosen value with the correct length(flat line on graph)
            plt.plot([x_choose]*len(x), (y), color="black")

            def GraphSetting():
                plt.suptitle(topaxis+"="+str(t[index])+"("+heldunit+")")
                c3.set(xlabel=xaxis+xunit)
                c3.set(ylabel=yaxis+yunit)
                c3.set(xlim=(xmin, xmax))
                c3.set(ylim=(ymin, ymax))

                c1.set_xlabel(yaxis+" "+"("+yunit+")")

                c1.set(title=xaxis+""+"("+xunit+")"+"="+str(x_choose))
                c1.set(xlim=(ymin, ymax))
                c1.set(ylabel=datavar)
                c1.set(ylim=(datamin, datamax))
                c2.set(xlabel=xaxis+" "+"("+xunit+")"+"c1")

                c2.set(title=yaxis+" "+"("+yunit+")"+"="+str(y_choose))

                c2.set(xlim=(xmin, xmax))
                c2.set(ylabel=datavar)
                c2.set(ylim=(datamin, datamax))
            GraphSetting()

        def animate(i):
            index = i
            plt.clf()
            global xani
            global yani
            global tani
            if(Nt > Nx and Nt > Ny):
                z = np.reshape(data, (Nt, Nx, Ny, Nshot), order="F")
            elif(Nt >= Nx and Ny > Nt):
                z = np.reshape(data, (Ny, Nt, Nx, Nshot))
            else:
                z = np.reshape(data, (Nx, Nt, Ny, Nshot))

            if(len(yani) > len(xani)):
                yani = np.linspace(np.min(yani), np.max(yani), len(xani))
                print("yup")
            else:
                xani = np.linspace(np.min(xani), np.max(xani), len(yani))
                print("nope")
            z = z[index, :, :, int(shot)]
            xdataind = (np.abs(xani-x_choose)).argmin()

            xdata = z[:, xdataind]
            ydataind = (np.abs(yani-y_choose)).argmin()
            ydata = z[ydataind, :]

            c1 = fig.add_subplot(gs[0, 0])  # creates 1D slices
            plt.plot(yani, xdata, color="black")

            c2 = fig.add_subplot(gs[1, 0])
            plt.plot(xani, ydata, color="black")

            c3 = fig.add_subplot(gs[:, 1])

            plt.axis('square')
            if(vectorimpose == True):
                plt.pcolormesh(x, y, z, shading="gouraud",
                               cmap='hot', vmin=datamin, vmax=datamax)  # heatmap
                cbar = plt.colorbar()
                cbar.set_label(datavar, rotation=270)
                [xgrad, ygrad] = np.gradient(z)
                plt.quiver(x, y, xgrad, ygrad, width=0.01)
            elif(vector == True):
                [xgrad, ygrad] = np.gradient(z)
                plt.quiver(x, y, xgrad, ygrad, width=0.01)
            else:
                plt.pcolormesh(x, y, z, shading="gouraud",
                               cmap='hot', vmin=datamin, vmax=datamax)  # heatmap
                cbar = plt.colorbar()
            cbar.set_label(datavar, rotation=270)

            # # Constants that show where we are on heatmap

            plt.plot((x),
                     [y_choose]*len(y), color="black")
            # # X/Y Slices, float array is simply making a line at the chosen value with the correct length(flat line on graph)
            plt.plot([x_choose]*len(x), (y), color="black")

            def GraphSetting():
                c3.set(xlabel=xaxis+xunit)
                plt.suptitle(topaxis+"="+str(t[index])+"("+heldunit+")")
                c3.set(ylabel=yaxis+yunit)
                c3.set(xlim=(xmin, xmax))
                c3.set(ylim=(ymin, ymax))

                c1.set_xlabel(yaxis+" "+"("+yunit+")")

                c1.set(title=xaxis+""+"("+xunit+")"+"="+str(x_choose))
                c1.set(xlim=(ymin, ymax))
                c1.set(ylabel=datavar)
                c1.set(ylim=(datamin, datamax))

                c2.set(xlabel=xaxis+" "+"("+xunit+")"+"c1")

                c2.set(title=yaxis+" "+"("+yunit+")"+"="+str(y_choose))

                c2.set(xlim=(xmin, xmax))
                c2.set(ylabel=datavar)
                c2.set(ylim=(datamin, datamax))

            GraphSetting()
        if(len(t) > 500):
            anim = animation.FuncAnimation(
                fig, animateinter, frames=int(len(t)/48), repeat=False)
            # saving function, first we render to screen
            anim.save('Animation.gif', writer=PillowWriter(fps=100))
            AnimationDisp(int(len(t)/48))
        else:
            anim = animation.FuncAnimation(
                fig, animate, frames=int(len(t)), repeat=False)
            # saving function, first we render to screen
            anim.save('Animation.gif', writer=PillowWriter(fps=3))
            AnimationDisp(int(len(t)))

    def Plotting():
        # A tk.DrawingArea.
        canvas1 = FigureCanvasTkAgg(fig1, master=graphheat)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=2, column=1)
        Tk.update(graphheat)  # For xchange length logic
        Label(graphheat, text="Change the held Variable" +
              topaxis).grid(row=0, column=1)
        holdchange = Scale(graphheat, orient=HORIZONTAL, resolution=deltat, from_=heldmin, to=heldmax, length=int(
            canvas1.get_tk_widget().winfo_width()))  # Makes slider size of graph
        holdchange.set(heldchoose)
        holdchange.grid(row=1, column=1)

        Label(graphheat, text="Vertical Label".replace(" ", " \n"),
              wraplength=1).grid(row=2, column=3)  # x=C
        xchange = Scale(graphheat, resolution=deltat, orient=VERTICAL, length=int(canvas1.get_tk_widget().winfo_height(
        )), from_=xmin, to=xmax)  # Because vertical line we use ymin and max
        xchange.set(x_choose)
        xchange.grid(row=2, column=2)
        Tk.update(graphheat)
        Label(graphheat, text="Change the Horizontal plot").grid(
            row=4, column=1)  # y=C
        ychange = Scale(graphheat, resolution=deltat, orient=HORIZONTAL, length=int(canvas1.get_tk_widget().winfo_width(
        )), from_=ymin, to=ymax)  # Because vertical line we use ymin and max
        ychange.set(y_choose)
        ychange.grid(row=3, column=1)
        Tk.update(graphheat)
        Button(graphheat, text="Re-render graphs",
               command=lambda: [ClearGraph(canvas1),
                                Graph_heatmap(xaxis, yaxis, topaxis, xmin, ymin, xmax, ymax, deltat, float(xchange.get()), float(ychange.get()), float(holdchange.get()), heldmax, heldmin, datavar, shot, vector, xunit, xfac, yunit, yfac, heldunit, heldfac, False, datamin, datamax, vectorimpose)]).grid(row=5, column=1)

        quit = Button(graphheat, text="Quit", command=graphheat.destroy)
        quit.grid(row=6, column=1)

        Label(graphheat, text="Create animation of response to held variable ").grid(  # x_choose, y_choose, heldchoose, xmin, xmax, ymin, ymax
            row=1, column=4)
        Button(graphheat, command=lambda: Animation(xchange.get(), ychange.get(), holdchange.get(), xmin, xmax, ymin, ymax)).grid(
            row=1, column=5)  # Animation call,

        Label(graphheat, text="Save this Figure").grid(row=2, column=4)
        Button(graphheat, command=lambda: plt.savefig("heatmap at"+topaxis+"="+str(heldchoose) +
                                                      ","+xaxis+"="+str(x_choose)+","+str(yaxis)+"="+str(y_choose)+".png")).grid(row=2, column=5)
    Plotting()


def ClearGraph(canvas1):
    try:
        canvas1.get_tk_widget().pack_forget()
    except AttributeError:
        pass


# file logic
def FileLoader():
    for widget in root.winfo_children():  # destroys everything in root thats already been called
        widget.destroy()
    Label(root, text="how many files would you like to load in(integer only)").pack()
    nfile = Entry(root, width=5)
    nfile.pack()

    Label(root, text="Number of trials(integer only)").pack()
    shots = Entry(root, width=5)
    shots.pack()
    Label(root, text="Go to File settings")
    Button(root, command=lambda: [FileWarning(
        int(nfile.get())), Getshots(shots.get()), Filename(int(nfile.get()))]).pack()

    def Getshots(shot):
        global Nshot
        Nshot = int(shot)

    def FileWarning(n):
        if(n > 2):
            pass
        else:
            messagebox.showwarning(
                title="Caution", message="2 or less files will result in no graph being made")


def Filename(n):
    filenames = []
    for widget in root.winfo_children():  # destroys everything in root thats already been called
        widget.destroy()
    for i in range(n):
        Label(root, text="name for file(including extension)").pack()
        filename = Entry(root, width=20)
        filename.pack()
        filenames.append(filename)
    Button(root, text="Continue", command=lambda: Columnsetting(
        n, GetNames(filenames))).pack()
    Button(root, text="Back", command=lambda: FileLoader()).pack()

    def GetNames(filenames):
        nnames = []
        for entries in filenames:
            nnames.append(entries.get())
        return nnames


def Columnsetting(n, nnames):  # prompts number of columns per file
    columns = []
    for widget in root.winfo_children():  # destroys everything in root thats already been called
        widget.destroy()
    for i in range(n):
        Label(root, text=(str(nnames[i]) +
                          ", Number of columns loading in")).pack()
        column = Entry(root, width=5)
        column.pack()
        columns.append(column)
    Label(root, text="Move to column settings").pack()

    def GetColumns(columns):
        column = []
        for entries in columns:
            column.append(entries.get())
        return column

    Button(root, command=lambda: ColumnName(n, GetColumns(
        columns), nnames)).pack()  # n is number of files
    Button(root, text="Back", command=lambda: Filename(n)).pack()


def ColumnName(nfiles, n, nnames):  # Prompts name of each column
    colnames = []
    colsum = 0
    for widget in root.winfo_children():  # destroys everything in root thats already been called
        widget.destroy()
    n = np.reshape(n, -1)  # reshaping for use in array
    n = np.array(n, dtype=int)
    for i in range(len(n)):
        for j in range(n[i]):
            Label(root, text="Name of Column " +
                  str(j+1)+" of"+str(nnames[i])).pack()  # +1 to account for counting
            colname = Entry(root, width=30)
            colname.pack()
            colsum = colsum+n[i]  # number of iterations for later
            colnames.append(colname)

    def GetCol(colnames):
        columns = []
        for entries in colnames:
            columns.append(entries.get())
        return columns

    # creates scheme to pass to data folder [[name,#,[names of columns]]]
    def Master(col):
        offset = 0
        master = []
        for i in range(nfiles):
            colpile = []
            if(n[i] != 1):
                for j in range(n[i]):
                    colpile.append(col[j])
                master.append([nnames[i], n[i], colpile])
                offset = offset+n[i]-1  # Because programmer indexing 0=1
            else:
                master.append([nnames[i], n[i], col[i+offset]])
        return master
    Button(root, text="back", command=lambda: Columnsetting(len(n), nnames)).pack()
    Button(root, text="Continue", command=lambda: Filereader(
        Master(GetCol(colnames)), nfiles)).pack()

    def Filereader(master, nfiles):  # backend, reads files

        global frame
        frame = pd.DataFrame(None)
        unconcancatatedframe = []

        def ExtensionReader():
            filename = tuple[0]
            # splits into [<file name>.<ext>]
            filesplit = os.path.splitext(filename)
            return filesplit[1]

        def ExtensionSelect():

            if(extension == '.csv'):  # csv reading method
                t = []  # temp variable
                t.append(tuple[2])
                t = np.reshape(t, -1)
                namecol = []
                for i in range(len(t)):
                    namecol.append(str(t[i]))
                data = pd.read_csv(str(tuple[0]), names=namecol)
                tempframe = pd.DataFrame(data)

            elif(extension == '.bin'):  # Binary reading method
                dt = np.dtype([('a', 'i4'), ('b', 'i4'), ('c', 'i4'), ('d', 'f4'),
                               ('e', 'i4'), ('f', 'i4', (256,))])  # Binary formatting

                data = np.fromfile(str(tuple[0]), dtype=dt)

                tempframe = pd.DataFrame.from_records(data, columns=list(
                    tuple[2]))  # Note columns may not work, test

            elif(extension == 'hdf5'):  # hdf5 reading
                data = pd.read_hdf(str(tuple[0], names=list(tuple[2])))
                tempframe = pd.DataFrame(data)

            else:  # Error handling
                FileLoader()  # calls back to begining
                messagebox.ERROR(message="Unsupported file type")

            unconcancatatedframe.append(tempframe)

        for i in range(nfiles):  # runs through all files
            tuple = master[i]  # temporary variable
            extension = ExtensionReader()
            ExtensionSelect()
        frame = pd.concat(unconcancatatedframe)

        for widget in root.winfo_children():  # destroys everything in root thats already been called
            widget.destroy()

        Graph_select()  # Calls method to select graph


FileLoader()
global update, updateframe
update = False
updateframe = False


def AnimationDisp(Framecnt):
    def aniwindowcreate():
        global aniwindow
        aniwindow = Toplevel()
        aniwindow.title("Animation window")
        aniwindow.geometry('1500x1000')
    aniwindowcreate()
    progress = IntVar()
    ttk.Progressbar(aniwindow, orient='horizontal',
                    mode='determinate', variable=progress).pack()
    Button(aniwindow, text="Pause unpause", command=lambda: [(
        changeupdate()), Tk.update(aniwindow)]).pack()

    Label(aniwindow, text="Select a frame")
    framevariable = IntVar(aniwindow)
    Scale(aniwindow, orient=HORIZONTAL, from_=1,
          to=21, variable=framevariable).pack()
    Button(
        aniwindow, text="Stop or start animation and go to select frame", command=lambda: [changeupdateframe(), Tk.update(aniwindow)]).pack()

    def displayanimation(frameCnt):
        frames = [PhotoImage(
            file='Animation.gif', format='gif -index %i' % (i)) for i in range(frameCnt)]

        def updates(ind):
            frame = frames[ind]
            ind += 1
            if ind == frameCnt:
                ind = 0
                progress.set(0)
            label.configure(image=frame)  # pb['value'] += 20
            progress.set(100*ind/(21))
            Tk.update(aniwindow)
            if(update == False and updateframe == False):

                aniwindow.update()
                aniwindow.after(300, updates, ind)
            elif(updateframe == True):

                # due to pythonic indexing
                frame = frames[framevariable.get()-1]
                label.configure(image=frame)
                # due to pythonic indexing
                progress.set(100/(21/(framevariable.get()+1)))

                while(updateframe == True):
                    Tk.update(aniwindow)
                    print(updateframe)
                    if(updateframe == False):
                        updates(ind)
                        print("success")

            elif(update == True):
                while(update == True):

                    Tk.update(aniwindow)
                    print(update)
                    if(update == False):

                        updates(ind)
                        print("success")

        label = Label(aniwindow)
        label.pack()
        aniwindow.after(0, updates, 0)
    displayanimation(Framecnt)


def changeupdate():
    global update
    update = not update


def changeupdateframe():
    global updateframe
    updateframe = not updateframe


def radiussettings():
    column_list = frame.columns.tolist()
    OPTIONS = column_list  # this is what solved my problem
    x_axis_var = StringVar(radiussetting)
    Label(radiussetting, text="Variable of x-axis").grid(row=0, column=0)
    OptionMenu(radiussetting, x_axis_var,
               *OPTIONS).grid(row=1, column=0)

    Label(radiussetting, text="Cross-sectional number for x-axis (must be between x min and max)").grid(row=2, column=0)
    x_choose = Entry(radiussetting, width=5)
    x_choose.grid(row=3, column=0)

    Label(radiussetting, text="x-axis min").grid(row=4, column=0)
    xmin = Entry(radiussetting, width=5)
    xmin.grid(row=5, column=0)

    Label(radiussetting, text="x-axis max").grid(row=7, column=0)
    xmax = Entry(radiussetting, width=5)
    xmax.grid(row=8, column=0)

    Label(radiussetting, text="x unit").grid(row=9, column=0)
    xunit = Entry(radiussetting, width=15)
    xunit.grid(row=10, column=0)

    Label(radiussetting, text="x unit factor(ex: cm->m= 100)").grid(row=11, column=0)
    xfac = Entry(radiussetting, width=5)
    xfac.grid(row=12, column=0)

    y_axis_var = StringVar(radiussetting)
    Label(radiussetting, text="Variable of heatmap y-axis").grid(row=0, column=1)
    OptionMenu(radiussetting, y_axis_var,
               *OPTIONS).grid(row=1, column=1)

    Label(radiussetting,
          text="Cross-sectional number for heatmap y-axis").grid(row=2, column=1)
    y_choose = Entry(radiussetting, width=5)
    y_choose.grid(row=3, column=1)

    Label(radiussetting, text=" heatmap ymin").grid(row=4, column=1)
    ymin = Entry(radiussetting, width=5)
    ymin.grid(row=6, column=1)

    Label(radiussetting, text="ymax").grid(row=7, column=1)
    ymax = Entry(radiussetting, width=5)
    ymax.grid(row=8, column=1)

    Label(radiussetting, text="y unit").grid(row=9, column=1)
    yunit = Entry(radiussetting, width=15)
    yunit.grid(row=10, column=1)

    Label(radiussetting, text="y unit factor(ex: cm->m= 100)").grid(row=11, column=1)
    yfac = Entry(radiussetting, width=5)
    yfac.grid(row=12, column=1)

    Label(radiussetting, text="Held Variable").grid(
        row=0, column=2)  # Thing we will be scrolling through
    held_var = StringVar(radiussetting)
    OptionMenu(radiussetting, held_var, *OPTIONS).grid(
        row=1, column=2)

    Label(radiussetting, text="Held Variable Min").grid(
        row=4, column=2)  # For scroll function
    held_choosemin = Entry(radiussetting, width=5)
    held_choosemin.grid(row=5, column=2)

    Label(radiussetting, text="Held Variable Max").grid(
        row=6, column=2)  # For scroll function
    held_choosemax = Entry(radiussetting, width=5)
    held_choosemax.grid(row=7, column=2)

    Label(radiussetting, text="held unit").grid(row=8, column=2)
    heldunit = Entry(radiussetting, width=15)
    heldunit.grid(row=9, column=2)

    Label(radiussetting, text="held unit factor(ex: cm->m= 100)").grid(row=10, column=2)
    heldfac = Entry(radiussetting, width=5)
    heldfac.grid(row=11, column=2)

    Label(radiussetting, text="Data Variable1").grid(
        row=0, column=5)  # For scroll function
    datavar1 = StringVar(radiussetting)
    OptionMenu(radiussetting, datavar1, *OPTIONS).grid(row=1, column=5)

    Label(radiussetting, text="Data Constant1(should be DC)").grid(
        row=2, column=5)  # For scroll function
    dataconstant1 = StringVar(radiussetting)
    OptionMenu(radiussetting, dataconstant1, *OPTIONS).grid(row=3, column=5)

    Label(radiussetting, text="Data Unit 1").grid(row=4, column=5)
    dataunit1 = Entry(radiussetting, width=15)
    dataunit1.grid(row=5, column=5)

    Label(radiussetting, text="Data unit 1 factor(ex: cm->m= 100)").grid(row=6, column=5)
    dataunit1fac = Entry(radiussetting, width=5)
    dataunit1fac.grid(row=7, column=5)

    Label(radiussetting, text="Data Variable2").grid(
        row=8, column=5)  # For scroll function
    datavar2 = StringVar(radiussetting)
    OptionMenu(radiussetting, datavar2, *OPTIONS).grid(row=9, column=5)

    Label(radiussetting, text="Data Constant2(should be DC)").grid(
        row=10, column=5)  # For scroll function
    dataconstant2 = StringVar(radiussetting)
    OptionMenu(radiussetting, dataconstant2, *OPTIONS).grid(row=11, column=5)

    Label(radiussetting, text="Data Unit 2").grid(row=12, column=5)
    dataunit2 = Entry(radiussetting, width=15)
    dataunit2.grid(row=13, column=5)

    Label(radiussetting,
          text="Data unit 2 factor(ex: cm->m= 100)").grid(row=14, column=5)
    dataunit2fac = Entry(radiussetting, width=5)
    dataunit2fac.grid(row=15, column=5)
    for i in range(len(OPTIONS)):
        maxmin = Label(radiussetting, text=(OPTIONS[i]+" "+"Has a max of"+" "+str(np.around((np.max(frame[OPTIONS[i]])), 4)) +
                                            " "+"and a min of"+" "+str(np.around(np.min(frame[OPTIONS[i]]), 4))))
        maxmin.grid(row=i, column=4)
    Label(radiussetting,
          text="what should the change in radius of the concentric circles be").grid(row=16, column=5)
    radius = Entry(radiussetting, width=5)
    radius.grid(row=17, column=5)
    Label(radiussetting, text='resolution').grid(row=18, column=5)
    resolution = Entry(radiussetting, width=5)
    resolution.grid(row=19, column=5)
    Label(radiussetting, text='len of radius').grid(
        row=20, column=5)
    length = Entry(radiussetting, width=5)
    length.grid(row=21, column=5)

    Label(radiussetting, text='offset of angle due to probe effects 180 (use degrees)').grid(
        row=22, column=5)
    offset = Entry(radiussetting, width=5)
    offset.grid(row=23, column=5)
    try:
        ind = column_list.index('x')
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        x_axis_var.set(OPTIONS[ind])
    except ValueError:
        pass
    try:
        ind = column_list.index('X')
        print(ind)
        print(OPTIONS)
        print(*OPTIONS)
        xminn, xmaxn, parmsx = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        x_axis_var.set(OPTIONS[ind])
    except ValueError:
        pass

    try:
        ind = column_list.index('y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        y_axis_var.set(OPTIONS[ind])
    except ValueError:
        pass
    try:
        ind = column_list.index('Y')
        yminn, ymaxn, parmsy = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        y_axis_var.set(OPTIONS[ind])
    except ValueError:
        pass

    try:
        ind = column_list.index('time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        held_var.set(OPTIONS[ind])
    except ValueError:
        pass
    try:
        ind = column_list.index('Time')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        held_var.set(OPTIONS[ind])
    except ValueError:
        pass

    try:
        ind = column_list.index('T')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        held_var.set(OPTIONS[ind])
    except ValueError:
        pass
    try:
        ind = column_list.index('t')
        tminn, tmaxn, parmst = np.around((np.min(frame[OPTIONS[ind]])), 4), np.around(
            (np.max(frame[OPTIONS[ind]])), 4), True
        held_var.set(OPTIONS[ind])
    except ValueError:
        pass
    try:
        if(parmst == True):
            held_choosemin.insert(END, tminn)
            held_choosemax.insert(END, tmaxn)
    except TypeError:
        pass
    heldfac.insert(END, 1)
    try:
        if(parmsx == True):
            xmin.insert(END, xminn)
            xmax.insert(END, xmaxn)
    except TypeError:
        pass
    xfac.insert(END, 1)
    try:
        if(parmsy == True):
            ymin.insert(END, yminn)
            ymax.insert(END, ymaxn)
    except TypeError:
        pass
    yfac.insert(END, 1)

    Button(radiussetting, text="plot graphs", command=lambda: [radiusgraphwindow(), radiusGraphs(
        x_axis_var.get(), y_axis_var.get(), held_var.get(), xfac.get(), yfac.get(), heldfac.get(
        ), xmin.get(), ymin.get(), xmax.get(), ymax.get(), x_choose.get(), y_choose.get(), 0, 0, 0, 0,
        datavar1.get(), dataconstant1.get(), datavar2.get(), dataconstant2.get(), True, held_choosemin.get(
        ), held_choosemax.get(), radius.get(), False, False, dataunit1fac.get(), dataunit2fac.get(),
        dataunit1.get(), dataunit2.get(), resolution.get(), xunit.get(), length.get(), offset.get(), False, 0, 0.001, -10, 10, heldunit.get())
    ]).grid(row=30, column=6)


def radiusGraphs(xaxis, yaxis, topaxis, xfac, yfac, heldfac, xmin, ymin, xmax, ymax, x_choose, y_choose, x, y, t, nplots, var1, var2, var3, var4, firstrender, heldmin, heldmax, deltar, centrechange, calculategraphs,
                 datafac1, datafac2, dataunit1, dataunit2, deltat, xunit, rgraphmin, offset, tablecommence, tablevalues, floor, datamin, datamax, heldunit):
    print(xfac, yfac, heldfac, xmin, ymin, xmax, ymax, x_choose,
          y_choose, deltar, heldmin, heldmax, deltat, rgraphmin, offset)
    print(floor)
    floor = float(floor)
    xfac, yfac, heldfac, xmin, ymin, xmax, ymax, x_choose, y_choose, deltar, heldmin, heldmax, deltat, rgraphmin, offset = \
        float(xfac), float(yfac), float(heldfac), float(xmin), float(ymin), float(xmax), float(ymax), float(x_choose), float(
            y_choose), float(deltar), float(heldmin), float(heldmax), float(deltat), float(rgraphmin), float(offset)
    datamin, datamax = float(datamin), float(datamax)
    offset = offset*np.pi/180  # converts angle to radians
    skewinputarray = []
    skewtotalarray = []
    pdfdefault = []
    xdefault = []
    skewdefault = []
    radiusdefault = []

    def centre(zheat):
        nonlocal x, y, x_choose, y_choose, xmin, ymin, xmax, ymax
        center = np.mean(np.argwhere(zheat > 0.9), axis=0)
        xc, yc = (x[int(np.fix(center[0]))
                    ]), y[int(np.fix((center[1])))]
        x = x-xc
        xmin = xmin-xc
        xmax = xmax-xc
        x_choose = x_choose-xc
        y = y-yc
        ymin = ymin-yc
        ymax = ymax-yc
        y_choose = y_choose-yc
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    if(tablecommence == False):
        gs = GridSpec(2, 3, wspace=0.5, hspace=0.6)
    else:
        gs = GridSpec(2, 6, wspace=0.5, hspace=0.6)
    if(firstrender == True):
        x = frame[[xaxis]].to_numpy()
        x = x[~np.isnan(x)]
        y = frame[[yaxis]].to_numpy()
        y = y[~np.isnan(y)]
        t = frame[[topaxis]].to_numpy()
        t = t[~np.isnan(t)]
        nplots = [[var1, var2], [var3, var4]]
        x = x*float(xfac)
        y = y*float(yfac)
        t = t*float(heldfac)
    dataplots = [dataunit1, dataunit2]
    Nt = len(t)
    Nx = len(x)
    Ny = len(y)

    def usercentre():
        nonlocal x, y, x_choose, y_choose, xmin, ymin, xmax, ymax
        xc = x_choose
        yc = y_choose
        x = x-xc
        xmin = xmin-xc
        xmax = xmax-xc
        x_choose = x_choose-xc
        y = y-yc
        ymin = ymin-yc
        ymax = ymax-yc
        y_choose = y_choose-yc
        return
    if(centrechange == True):
        usercentre()

    def CreatePlot():
        print('start')

        def radius():
            r = []
            for i in range(len(x)):
                for j in range(len(y)):
                    r.append(
                        [np.abs(np.sqrt(x[i]**2+y[j]**2)), i, j])  # gives all radi and their postions in the x y arrays
            return r
        skewkurtarray = []

        def skewkurt():
            nonlocal skewdefault
            if(tablecommence == True):
                length = 4
                c1 = fig.add_subplot(gs[a, 5])
            else:
                length = 1
                c1 = fig.add_subplot(gs[a, 2])
            nonlocal skewinputarray, skewkurtarray
            markers = ['*', 'D', 'P', 'o']
            skewinputarray = np.reshape(
                skewinputarray, (length, len(radiuschange)-1, 2))
            skewdefault.append(skewinputarray)
            colors = cm.cool(np.linspace(0, 1, len(radiuschange)-1))
            for j in range(length):
                xtotal = []
                ytotal = []
                # doing skewinputarray[j][i][#] wasnt working for some reason so split it into 2 and works now
                array = skewinputarray[j]
                for i in range(len(radiuschange)-1):
                    x_scatter = array[i][0]
                    y_scatter = array[i][1]
                    if(j == 0):
                        radiuschangearray = np.linspace(
                            np.min(radiuschange), np.max(radiuschange), len(radiuschange)-1)
                        plt.scatter(
                            x_scatter, y_scatter, color=colors[i], marker=markers[j],
                            # label=('radius'+str(np.around(
                            # radiuschangearray[i], 2))+xunit) too big the issue
                        )
                    else:
                        plt.scatter(
                            x_scatter, y_scatter, color=colors[i], marker=markers[j])
                    xtotal.append(x_scatter)
                    ytotal.append(y_scatter)
                p = np.polyfit(
                    np.reshape(xtotal, -1), np.reshape(ytotal, -1), 2)
                skewline = np.linspace(
                    np.min(np.min(xtotal)), np.max(xtotal), len(xtotal))
                yfit = np.polyval(p, skewline)
                line, = plt.plot(skewline, yfit, label='graph' + str(j+1),
                                 linestyle='dashed', marker=markers[j], markevery=10000)
            skewinputarray = []
            skewkurtarray = []
            legend = plt.legend(fontsize='xx-small')
            c1.set(ylabel='kurtosis')
            c1.set(xlabel='skew')
        for a in range(len(nplots)):

            data = frame[[nplots[a][0]]].to_numpy()
            data = data[~np.isnan(data)]
            dataconstant = frame[[nplots[a][1]]].to_numpy()
            dataconstant = dataconstant[~np.isnan(dataconstant)]

            ztimeseries = np.reshape(
                data, (Nt, Nx, Ny, Nshot), order="F")
            zconstant = np.reshape(
                dataconstant, (Nt, Nx, Ny, Nshot), order="F")
            zheat = np.reshape(
                dataconstant, (Nt, Nx, Ny, Nshot), order="F")  # for heatmap
            zheat = np.mean(zheat, axis=-1)

            if(a == 0 and firstrender == True):
                centre(zheat[0, :, :])
            radiusarray = radius()

            def heatmap():
                nonlocal zheat
                zheat = zheat[0, :, :]
                c2 = fig.add_subplot(gs[a, 0])
                plt.pcolormesh(x, y, zheat, shading='gouraud',
                               vmin=np.min(zheat), vmax=np.max(zheat), cmap='hot')
                plt.plot((x),
                         [y_choose]*len(y), color="black")
                # # X/Y Slices, float array is simply making a line at the chosen value with the correct length(flat line on graph)
                plt.plot([x_choose]*len(x), (y), color="black")
                c2.set(xlabel=xaxis+"("+xunit+")")
                c2.set(ylabel=yaxis+"("+xunit+")")
                c2.set(title=topaxis+'=0')  # heatmap
                cbar = plt.colorbar()
                cbar.ax.set_title(
                    str(nplots[a][1])+'('+dataplots[a]+')')
                rheat = np.linspace(0, rgraphmin, 20)
                xru = rheat*np.cos(offset+np.pi)
                yru = rheat*np.sin(np.pi+offset)
                plt.plot(xru, yru, color='teal')
                xrl = rheat*np.cos(-offset+np.pi)
                yrl = rheat*np.sin(np.pi-offset)
                thetaline = np.linspace(0, 2*np.pi, 200)
                plt.plot(xrl, yrl, color='teal')
                xcir = rgraphmin*np.cos(thetaline)
                ycir = rgraphmin*np.sin(thetaline)
                plt.plot(xcir, ycir, color='dodgerblue')
            heatmap()

            def azaveraged():
                zaz = []
                radiuschange = np.linspace(
                    0, rgraphmin, int(np.fix(rgraphmin/deltar)))
                for i in range(Nt):
                    if(t[i] >= heldmin and heldmax >= t[i]):
                        for k in range(len(radiuschange)-1):
                            min = radiuschange[k]
                            max = radiuschange[k+1]
                            for l in range(len(radiusarray)):
                                if(radiusarray[l][0] >= min and radiusarray[l][0] <= max):
                                    theta = np.arctan2(
                                        y[radiusarray[l][2]], x[radiusarray[l][1]])
                                    if(theta < 0):
                                        theta = 2*np.pi+theta
                                    if(theta > np.pi-offset and theta < np.pi+offset):
                                        pass
                                    else:
                                        for j in range(Nshot):
                                            zaz.append([(
                                                ztimeseries[i, [radiusarray[l][1]], radiusarray[l][2], j]/zconstant[i, [radiusarray[l][1]], radiusarray[l][2], j]), k])

                return zaz, radiuschange

            def graphinglogicax(zaz):
                nonlocal radiusdefault, xdefault
                levelcurve = []
                raw = []
                extremeradii = []

                def levelcurvecreator():
                    nonlocal levelcurve
                    for i in range(len(radiuschange)-1):
                        temp = []
                        for j in range(len(zaz)):
                            if(zaz[j][1] == i):
                                temp.append(zaz[j][0])
                        temp = temp/np.std(temp)
                        levelcurve.append(temp)
                    levelcurve = np.asarray(levelcurve)
                levelcurvecreator()

                maxes = []

                def rawmaker(array):
                    nonlocal maxes
                    for i in range(len(array)):
                        for j in range(len(array[i])):
                            raw.append(array[i][j])
                    maxes = [np.min(raw), np.max(raw)]
                # shows the y slices, the peaks are ALWAYS at x=0, so on the heatmap,

                def binschecker(bins, hist, array):
                    jindicies = []
                    level = array

                    def arraychecker():
                        temp = []
                        mask = np.where((floor < np.abs(array)),
                                        array, np.nan)
                        temp = mask[~np.isnan(mask)]
                        print(np.shape(array))
                        print('divide')
                        print(np.shape(temp))
                        return temp

                    for j in range(len(bins)):
                        if(bins[j] < float(floor)):
                            jindicies.append(j)
                    bins = np.delete(bins, jindicies)
                    hist = np.delete(hist, jindicies)
                    hist = np.delete(hist, -1)
                    hist = np.reshape(hist, -1)
                    level = arraychecker()

                    return bins, hist, level

                def skewkurtosisfinder():
                    levelcurvemod = levelcurve
                    for i in range(len(radiuschange)-1):
                        bins1, hist1 = np.histogram(
                            levelcurve[i], density=True, bins=500)
                        bins, hist, levelcurvemod[i] = binschecker(
                            bins1, hist1, levelcurve[i])

                        skewinputarray.append(
                            [skew(levelcurvemod[i]), kurtosis(levelcurvemod[i], fisher=True)])
                    return levelcurvemod
                levelcurvemod = skewkurtosisfinder()
                rawmaker(levelcurvemod)
                pdfexport = []
                lengths = []
                averages = []
                pdf = []
                argmeans = []

                def appender():
                    nonlocal pdf, lengths, averages, argmeans
                    for i in range(len(radiuschange)-1):
                        Nbin = 500
                        max = np.max(np.abs(maxes))
                        xcomparison = np.linspace(-max, max, Nbin)
                        marking = 0

                        def arrayshortening(comparisonarray, array):
                            marker = 0
                            for j in range(len(comparisonarray)):
                                if(comparisonarray[j] < np.min(array) or comparisonarray[j] > np.max(array)):
                                    marker = marker+1
                            return marker
                        marking = arrayshortening(xcomparison, levelcurve[i])
                        binsmod, binsedge = np.histogram(
                            levelcurve[i], density=True, bins=Nbin-marking)

                        binsmod, binsedge, levelcurve[i] = binschecker(
                            binsmod, binsedge, levelcurve[i])

                        binsline, ignored = np.histogram(
                            levelcurve[i], density=True, bins=Nbin)
                        xline = np.linspace(
                            np.min(levelcurve[i]), np.max(levelcurve[i]), Nbin)
                        maxind = np.argmax(binsline)
                        argmean = np.argmax(binsmod)

                        def centering(array, selectedvalue):
                            left = selectedvalue
                            right = np.abs((len(array)-1)-selectedvalue)
                            toadd = np.abs(left-right)
                            if(left < right):
                                array = np.insert(array, 0, np.zeros(toadd))
                            if(right < left):
                                array = np.concatenate(
                                    (array, np.zeros(toadd)))
                            return array

                        bins = centering(binsmod, argmean)
                        pdf.append(bins)
                        lengths.append(len(bins))
                        averages.append(xline[maxind])
                        argmeans.append(argmean)
                appender()

                def lengthss():
                    maxlength = np.max(lengths)
                    if(maxlength % 2 != 0):
                        maxlength = maxlength+1
                    return maxlength
                maxlength = lengthss()

                def filling(array, selectedlength):
                    while(len(array) < selectedlength):
                        if(len(array) % 2 == 0):
                            array = np.insert(array, 0, 0)
                            array = np.concatenate((array, np.zeros(1)))
                        else:
                            array = np.insert(array, 0, 0)
                    return array
                for i in range(len(radiuschange)-1):
                    pdf[i] = filling(pdf[i], maxlength)
                xgraph = np.linspace(maxes[0], maxes[1], maxlength)
                for i in range(len(averages)):
                    averages[i] = np.argmin(np.abs(xgraph-averages[i]))

                def finalconditioning():
                    for i in range(len(radiuschange)-1):
                        if(np.argmax(pdf[i]) > averages[i]):
                            pdfindex = np.argmax(pdf[i])
                            xindex = averages[i]
                            subtraction = pdfindex-xindex
                            pdf[i] = np.delete(pdf[i], np.linspace(
                                0, (subtraction-1), (subtraction), dtype='int'))
                            pdf[i] = np.concatenate((pdf[i], np.zeros(
                                (subtraction))))
                        elif(np.argmax(pdf[i]) < averages[i]):
                            pdfindex = np.argmax(pdf[i])
                            xindex = averages[i]
                            subtraction = pdfindex-xindex
                            pdf[i] = np.delete(pdf[i], np.linspace(
                                len(pdf[i])+subtraction, len(pdf[i])-1, (np.abs(subtraction)), dtype='int'))
                            pdf[i] = np.insert(
                                pdf[i], 0, np.zeros(np.abs(subtraction)))
                        else:
                            pass
                    return pdf
                pdf = finalconditioning()

                def centreing():
                    nonlocal xgraph, pdfexport
                    if(np.abs(maxes[0]) != np.max(np.abs(maxes))):
                        xgraph = np.insert(xgraph, 0, -1*maxes[1])
                        for i in range(len(radiuschange)-1):
                            temp = pdf[i]
                            temp = np.insert(temp, 0, np.zeros(1))
                            pdfexport.append(temp)
                    elif(np.abs(maxes[1]) != np.max(np.abs(maxes))):
                        xgraph = np.concatenate(
                            (xgraph, [-1*maxes[0]]))
                        for i in range(len(radiuschange)-1):
                            temp = pdf[i]
                            temp = np.concatenate((temp, np.zeros(1)))
                            pdfexport.append(temp)

                    else:
                        pdfexport = pdf
                    return xgraph, pdfexport
                xgraph, pdfexport = centreing()
                return xgraph, pdfexport

            def graphingaz(xgraph, radiuschange, modpdf, i):
                radiusgraph = np.linspace(
                    np.min(radiuschange), np.max(radiuschange), len(radiuschange)-1)
                c1 = fig.add_subplot(gs[a, 1+i])
                withoutzeros = np.reshape(modpdf, -1)
                withoutzeros = withoutzeros[withoutzeros != 0]
                norms = mcolors.LogNorm(
                    np.min(withoutzeros), np.max(withoutzeros))
                im = plt.pcolormesh(xgraph, radiusgraph,
                                    modpdf, norm=norms, cmap='jet', rasterized=True)
                im.set_clim(floor, 10)
                plt.contour(xgraph, radiusgraph, modpdf, norm=norms,  # np.delete(pdf[i],0,np.zeros(index[i]-np.max[i]), np.concentrate((pdf[i],np.zeros(index[i]-np.max[i])))
                            colors=['black'])
                locator = LogLocator()
                formatter = LogFormatter()

                cbar = fig.colorbar(im, norm=norms)
                cbar.locator = locator
                cbar.formatter = formatter
                cbar.update_normal(im)
                cbar.ax.set_title('log10[PDF]')
                c1.set(
                    xlabel="$\\frac{\\delta"+" "+nplots[a][0]+"}{"+nplots[a][1]+"\\times \\sigma"+"}"+"$")
                c1.set(ylabel="radius"+'('+xunit+')')
                c1.set(xlim=(datamin, datamax))
                c1.set(title=str(heldmin)+'to'+str(heldmax)+'('+heldunit+')')
                pdfdefault.append(modpdf)
                return xgraph, radiusgraph
            nonlocal radiusdefault, xdefault
            if(calculategraphs == True):
                if(tablecommence == False):
                    azdata, radiuschange = azaveraged()
                    azmod, azpdf = graphinglogicax(azdata)
                    xgraph, radiusgraph = graphingaz(
                        azmod, radiuschange, azpdf, 0)
                    skewkurt()
                    radiusdefault = radiusgraph
                    xdefault.append(azmod)
                else:
                    for i in range(len(tablevalues)):
                        nonlocal heldmin, heldmax
                        heldmin = float(tablevalues[i][0])
                        heldmax = float(tablevalues[i][1])
                        azdata, radiuschange = azaveraged()
                        azmod, azpdf = graphinglogicax(azdata)
                        xgraph, radiusgraph = graphingaz(
                            azmod, radiuschange, azpdf, i)
                        xdefault.append(azmod)
                    skewkurt()
                    radiusdefault = radiusgraph

        return fig
    fig = CreatePlot()

    def Plotting(fig):
        canvas1 = FigureCanvasTkAgg(fig, master=radiusgraph)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=2, column=1)
        Tk.update(radiusgraph)  # For xchange length logic

        Label(radiusgraph, text="change vertical plot".replace(" ", " \n"),
              wraplength=1).grid(row=2, column=3)  # x=C
        xchange = Scale(radiusgraph, resolution=deltat, orient=VERTICAL, length=int(canvas1.get_tk_widget().winfo_height(
        )), from_=xmin, to=xmax)  # Because vertical line we use ymin and max
        xchange.set(x_choose)
        xchange.grid(row=2, column=2)
        Tk.update(radiusgraph)
        Label(radiusgraph, text="Change the Horizontal plot").grid(
            row=4, column=1)  # y=C
        ychange = Scale(radiusgraph, resolution=deltat, orient=HORIZONTAL, length=int(canvas1.get_tk_widget().winfo_width(
        )), from_=ymin, to=ymax)  # Because vertical line we use ymin and max
        ychange.set(y_choose)
        ychange.grid(row=3, column=1)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="xmin").grid(row=0, column=5)
        xminchange = Entry(radiusgraph, width=5)
        xminchange.insert(END, xmin)
        xminchange.grid(row=0, column=6)
        Tk.update(radiusgraph)
        Label(radiusgraph, text="xmax").grid(row=1, column=5)
        xmaxchange = Entry(radiusgraph, width=5)
        xmaxchange.insert(END, xmax)
        xmaxchange.grid(row=1, column=6)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="ymin").grid(row=0, column=7)
        yminchange = Entry(radiusgraph, width=5)
        yminchange.insert(END, ymin)
        yminchange.grid(row=0, column=8)
        Tk.update(radiusgraph)
        Label(radiusgraph, text="max").grid(row=1, column=7)
        ymaxchange = xmin
        ymaxchange = Entry(radiusgraph, width=5)
        ymaxchange.insert(END, ymax)
        ymaxchange.grid(row=1, column=8)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="heldmin").grid(row=0, column=9)
        heldminchange = Entry(radiusgraph, width=5)
        heldminchange.insert(END, heldmin)
        heldminchange.grid(row=0, column=10)
        Tk.update(radiusgraph)
        Label(radiusgraph, text="heldmax").grid(row=1, column=9)
        heldmaxchange = xmin
        heldmaxchange = Entry(radiusgraph, width=5)
        heldmaxchange.insert(END, heldmax)
        heldmaxchange.grid(row=1, column=10)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="x-axis min graph").grid(row=2, column=9)
        dataminchange = Entry(radiusgraph, width=5)
        dataminchange.insert(END, datamin)
        dataminchange.grid(row=2, column=10)
        Tk.update(radiusgraph)
        Label(radiusgraph, text="y-axis min graph").grid(row=3, column=9)
        dataaxchange = xmin
        datamaxchange = Entry(radiusgraph, width=5)
        datamaxchange.insert(END, datamax)
        datamaxchange.grid(row=3, column=10)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="radius change").grid(row=0, column=11)
        deltarchange = xmin
        deltarchange = Entry(radiusgraph, width=5)
        deltarchange.insert(END, deltar)
        deltarchange.grid(row=0, column=12)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="len of radius").grid(row=1, column=11)
        rmingraphchange = xmin
        rmingraphchange = Entry(radiusgraph, width=15)
        rmingraphchange.insert(END, rgraphmin)
        rmingraphchange.grid(row=1, column=12)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="angle").grid(row=0, column=13)
        offsetchange = Entry(radiusgraph, width=15)
        offsetchange.insert(END, np.around(offset*180/np.pi, 4))
        offsetchange.grid(row=0, column=14)
        Tk.update(radiusgraph)

        Label(radiusgraph, text="floor").grid(row=1, column=15)
        floorchange = Entry(radiusgraph, width=15)
        floorchange.insert(END, floor)
        floorchange.grid(row=1, column=16)
        Tk.update(radiusgraph)

        changedisplacement = BooleanVar(radiusgraph)
        Checkbutton(radiusgraph, text='crosshair becomes origin if true',
                    variable=changedisplacement).grid(row=3, column=5)

        commence = BooleanVar(radiusgraph)
        Checkbutton(radiusgraph, text='if true, render radius graphs (takes several minutes)',
                    variable=commence).grid(row=4, column=5)

        tablecommence = BooleanVar(radiusgraph)
        Checkbutton(radiusgraph, text='if true, render 2x2 table of graphs with 4 different time intervals(must select render as well)',
                    variable=tablecommence).grid(row=5, column=5)

        Label(radiusgraph, text="start of time interval 1").grid(row=3, column=6)
        table1min = Entry(radiusgraph, width=5)
        table1min.grid(column=7, row=3)

        Label(radiusgraph, text="end of time interval 1").grid(row=4, column=6)
        table1max = Entry(radiusgraph, width=5)
        table1max.grid(column=7, row=4)

        Label(radiusgraph, text="start of time interval 2").grid(row=5, column=6)
        table2min = Entry(radiusgraph, width=5)
        table2min.grid(column=7, row=5)

        Label(radiusgraph, text="end of time interval 2").grid(row=6, column=6)
        table2max = Entry(radiusgraph, width=5)
        table2max.grid(column=7, row=6)

        Label(radiusgraph, text="start of time interval 3").grid(row=7, column=6)
        table3min = Entry(radiusgraph, width=5)
        table3min.grid(column=7, row=7)

        Label(radiusgraph, text="end of time interval 3").grid(row=8, column=6)
        table3max = Entry(radiusgraph, width=5)
        table3max.grid(column=7, row=8)

        Label(radiusgraph, text="start of time interval 4").grid(row=9, column=6)
        table4min = Entry(radiusgraph, width=5)
        table4min.grid(column=7, row=9)

        Label(radiusgraph, text="end of time interval 4").grid(row=10, column=6)
        table4max = Entry(radiusgraph, width=5)
        table4max.grid(column=7, row=10)

        Button(radiusgraph, text="Re-render graphs",
               command=lambda: [ClearGraph(canvas1),
                                radiusGraphs(xaxis, yaxis, topaxis, xfac, yfac, heldfac, xminchange.get(), yminchange.get(), xmaxchange.get(), ymaxchange.get(), xchange.get(), ychange.get(),
                                             x, y, t, nplots, var1, var2, var3, var4,
                                             False, heldminchange.get(), heldmaxchange.get(
                                ), deltarchange.get(), changedisplacement.get(), commence.get(),
                   datafac1, datafac2, dataunit1, dataunit2, deltat, xunit, rmingraphchange.get(), offsetchange.get(), tablecommence.get(), [
                       [table1min.get(), table1max.get()], [table2min.get(), table2max.get()], [
                           table3min.get(), table3max.get()],
                       [table4min.get(), table4max.get()]], floorchange.get(), dataminchange.get(), datamaxchange.get(), heldunit)]).grid(row=8, column=1)

        quit = Button(radiusgraph, text="Quit", command=radiusgraph.destroy)
        quit.grid(row=9, column=1)
        Label(radiusgraph, text="Save this Figure").grid(row=2, column=4)
        Button(radiusgraph, command=lambda: [fig.set_size_inches(20, 10), plt.savefig("radius graphs"+"@" +
                                                                                      str(deltar)+".png"), fig.set_size_inches(10, 5)]).grid(row=2, column=5)
        Button(radiusgraph, text='DRAE', command=lambda: [Draeinitial(), draeviewer(
            skewdefault, pdfdefault, xdefault, radiusdefault, floor, True, None, None)]).grid(row=10, column=1)
    Plotting(fig)


def draeviewer(skewdefault, pdfdefault, xdefault, radiusdefault, floor, firstrender, indexremoved, removed):  # check if
    if(firstrender == True):
        removed = []
    if(firstrender == False):
        if(indexremoved != None):
            removed.append(indexremoved)

    def tablecondition():
        if(len(pdfdefault) == 2):
            nontable = True
        else:
            nontable = False
        return nontable

    def singlegraph(gs, fig):
        for i in range(2):
            c1 = fig.add_subplot(gs[i, 0])
            norms = mcolors.LogNorm(
                np.min(pdfdefault[i]), np.max(pdfdefault[i]))

            if(indexremoved != None):
                for j in range(len(removed)):
                    print(removed[j])
                    print(radiusdefault[removed[j]])
                    plt.plot(xdefault[i], [
                        radiusdefault[removed[j]]]*len(xdefault[i]), lw=5)
            im = plt.pcolormesh(xdefault[i], radiusdefault,
                                pdfdefault[i], norm=norms, cmap='jet', shading='gouraud', rasterized=True)
            im.set_clim(floor, 1)
            plt.contour(xdefault[i], radiusdefault, pdfdefault[i], norm=norms,  # np.delete(pdf[i],0,np.zeros(index[i]-np.max[i]), np.concentrate((pdf[i],np.zeros(index[i]-np.max[i])))
                        colors=['black'])
            locator = LogLocator()
            formatter = LogFormatter()

            cbar = fig.colorbar(im, norm=norms)
            cbar.locator = locator
            cbar.formatter = formatter
            cbar.update_normal(im)
            if(firstrender == True):
                pass
            else:
                pass

            def skewkurtosis():
                markers = ['*', 'D', 'P', 'o']
                colors = cm.cool(np.linspace(0, 1, len(radiusdefault)))
                for i in range(2):
                    data = skewdefault[i][0]
                    c2 = fig.add_subplot(gs[i, 1])
                    xtotal = []
                    ytotal = []
                    for j in range(len(radiusdefault)):
                        if(True == np.any(np.equal(j, removed))):
                            print(j)
                            pass
                        else:
                            x_scatter = data[j][0]
                            y_scatter = data[j][1]
                            plt.scatter(
                                x_scatter, y_scatter, color=colors[j], marker=markers[i])
                            plt.text(x_scatter+.03, y_scatter+.03,
                                     np.around(radiusdefault[j], 3), fontsize=8)
                            xtotal.append(x_scatter)
                            ytotal.append(y_scatter)
                    p = np.polyfit(
                        np.reshape(xtotal, -1), np.reshape(ytotal, -1), 2)
                    skewline = np.linspace(
                        np.min(np.min(xtotal)), np.max(xtotal), len(xtotal))
                    yfit = np.polyval(p, skewline)
                    line, = plt.plot(skewline, yfit,
                                     linestyle='dashed', marker=markers[i], markevery=10000,
                                     label='graph' + ''+str((i+1))+''+';'+'' +
                                     'a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                     str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3)))
                    plt.legend()
        skewkurtosis()

    def tablegraph(gs, fig):

        def skewkurtosis():
            markers = ['*', 'D', 'P', 'o']
            colors = cm.cool(np.linspace(0, 1, len(radiusdefault)))
            for k in range(2):
                xtotal = []
                ytotal = []
                for i in range(4):
                    if(i == 0):
                        data = skewdefault[i][0]
                        c2 = fig.add_subplot(gs[k, 4])
                        for j in range(len(radiusdefault)):
                            if(True == np.any(np.equal(j, removed))):
                                pass
                            else:
                                x_scatter = data[j][0]
                                y_scatter = data[j][1]
                                plt.scatter(
                                    x_scatter, y_scatter, color=colors[j], marker=markers[i])
                                plt.text(x_scatter+.03, y_scatter+.03,
                                         np.around(radiusdefault[j], 3), fontsize=6)
                                xtotal.append(x_scatter)
                                ytotal.append(y_scatter)
                        p = np.polyfit(
                            np.reshape(xtotal, -1), np.reshape(ytotal, -1), 2)
                        skewline = np.linspace(
                            np.min(np.min(xtotal)), np.max(xtotal), len(xtotal))
                        yfit = np.polyval(p, skewline)
                        line, = plt.plot(skewline, yfit,
                                         linestyle='dashed', marker=markers[i], markevery=10000, label='graph' + ''+str((i+1))+''+';'+'' +
                                         'a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                         str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3)))
                    else:
                        data = skewdefault[i+4][0]
                        c2 = fig.add_subplot(gs[k, 4])
                        for j in range(len(radiusdefault)):
                            x_scatter = data[j][0]
                            y_scatter = data[j][1]
                            plt.scatter(
                                x_scatter, y_scatter, color=colors[j], marker=markers[i])
                            plt.text(x_scatter+.03, y_scatter+.03,
                                     np.around(radiusdefault[j], 3), fontsize=6)
                            xtotal.append(x_scatter)
                            ytotal.append(y_scatter)
                        p = np.polyfit(
                            np.reshape(xtotal, -1), np.reshape(ytotal, -1), 2)
                        skewline = np.linspace(
                            np.min(np.min(xtotal)), np.max(xtotal), len(xtotal))
                        yfit = np.polyval(p, skewline)
                        line, = plt.plot(skewline, yfit,
                                         linestyle='dashed', marker=markers[i], markevery=10000, label='graph' + ''+str((i+1))+''+';'+'' +
                                         'a='+str(np.around(p[0], 3))+"," + ' '+'b=' +
                                         str(np.around(p[1], 3))+','+' '+'c='+str(np.around(p[2], 3)))
        for i in range(2):
            for j in range(4):
                c1 = fig.add_subplot(gs[i, j])
                if(i == 1):
                    norms = mcolors.LogNorm(
                        np.min(pdfdefault[j+4]), np.max(pdfdefault[j+4]))
                    plt.pcolormesh(
                        xdefault[j+4], radiusdefault, pdfdefault[j+4], norm=norms, cmap='jet')
                else:
                    norms = mcolors.LogNorm(
                        np.min(pdfdefault[j]), np.max(pdfdefault[j]))
                    plt.pcolormesh(
                        xdefault[j], radiusdefault, pdfdefault[j], norm=norms, cmap='jet')
        skewkurtosis()

    def graph(nontable):
        plt.clf()
        fig = plt.figure(figsize=(10, 5))
        print('figure')
        if(nontable == True):
            gs = GridSpec(2, 2, wspace=0.5, hspace=0.6)
            singlegraph(gs, fig)
        else:
            gs = GridSpec(2, 5, wspace=0.5, hspace=0.6)
            tablegraph(gs, fig)

        return fig

    def UI():
        def deletearray(indexremoved):
            nonlocal removed
            for i in range(len(removed)):
                if(indexremoved == removed[i]):
                    removed = np.delete(removed, i)
                    removed = removed.tolist()
                    break
            print(removed)
        Label(drae, text='Click checkbox to remove radii from graph').grid(
            row=0, column=1)
        canvas1 = FigureCanvasTkAgg(fig, master=drae)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=1, column=2)
        for i in range(len(radiusdefault)):
            print(i)
            print(np.any(np.equal(i, removed)))
            if(True == np.any(np.equal(i, removed))):
                Checkbutton(drae, variable=True, text=str(np.around(radiusdefault[i], 3)), command=lambda index=i: [deletearray(index),
                                                                                                                    draeviewer(skewdefault, pdfdefault, xdefault, radiusdefault, floor, False, None, removed)]).grid(row=i+1, column=0)

            else:
                Checkbutton(drae, variable=False, text=str(np.around(radiusdefault[i], 3)), command=lambda index=i: [
                            draeviewer(skewdefault, pdfdefault, xdefault, radiusdefault, floor, False, index, removed)]).grid(row=i+1, column=0)

        Button(drae, text='re-render', command=lambda: [draeviewer(skewdefault, pdfdefault,
                                                                   xdefault, radiusdefault, floor, False, None, removed)]).grid(row=0, column=2)
    table = tablecondition()
    fig = graph(table)
    UI()


def Draeinitial():
    global drae
    drae = Toplevel()
    drae.title('drae')
    drae.geometry('1920x1080')


def radiusgraphwindow():
    global radiusgraph
    radiusgraph = Toplevel()
    radiusgraph.title('radiusgraphs')
    radiusgraph.geometry('1920x1080')


root.mainloop()
