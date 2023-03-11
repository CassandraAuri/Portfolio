# Cassandra
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import emcee
from scipy.optimize import minimize
from IPython.display import display, Math
# import file, note may have to change import path as for me it was locally Explore/<>
# plt.style.use("cyberpunk")  # Dark mode!

labels = ["m", "b", "Variance"]
def Main(redshift, array):

    def Uncertainity_Calculations(x, y):
        x0 = np.linspace(-1.5, 3, 300)
        # Define matrix Z with shape 2,2421 : each column is a separate column vector [x_i, y_i]

        # Define the unit vector orthogonal to the defined line: Hogg+ Eq. 29

        z = np.matrix([x[0], y[0]])

        def MCMC():  # index
            def Orthogonal_Displacements(m, b): #Logic
                # Define matrix Delta (Hogg+ Eq. 30) with shape 1,2421 consisting of the
                #  orthogonal displacements of each data point x_i,y_i from the defined line
                vee = (np.matrix([-m, 1]).T / np.sqrt(1+m**2.))
                Delta = vee.T * z - b / np.sqrt(1+m**2.)
                return np.square(Delta.T)

            def Orthogonal_Variance(m): #Logic 
                # returns an array with the same shape as the x_errors
                zero = np.zeros_like(x[1])
                part1 = (np.array([x[1], zero, zero, y[1]]).T).reshape(
                    len(x[0]), 2, 2)  # Creates the 2x2 covariance matrix
                vee = (np.matrix([-m, 1]).T / np.sqrt(1+m**2.))
                Sigma1 = np.matmul(part1, vee)  # Covariance *v
                # print(Sigma1)
                # "squares" our covariance
                Sigma2 = np.matmul(Sigma1, vee)
                return Sigma2

            def log_likelihood(theta): #Logic
                m, b, v = theta
                deltasquared = Orthogonal_Displacements(m, b)
                sigmasquared = Orthogonal_Variance(m)
                LnlVariance = -np.sum(1/2*np.log(sigmasquared+v**2)) - \
                    np.sum((deltasquared/(2*(sigmasquared+v**2))))  # logic
                return LnlVariance
                #model=m*x[0]+b
                #return -0.5 * np.sum((y[0] - model) ** 2 / (y[1]**2+v**2))

            def log_prior(theta): #Condtions of variables
                m, b, v = theta  # variables being guessed
                if 0.3 < m < 2 and -10 < b < 10 and 0<v<10:  # conditions
                    return 0.0
                return -np.inf

            def log_probability(theta): #Condition Function
                lp = log_prior(theta)  # Call condition setter
                # if priors are false, make blah+lp=-inf, aka nothing
                if not np.isfinite(lp):
                    return -np.inf
                blah = log_likelihood(theta)
                return lp + blah

            def runner(): #Runs MCMC
                pos = np.random.rand(30, len(labels))  # randomizes starting position
                nwalkers, ndim = pos.shape

                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability,
                )  # ensemble initalization
                sampler.run_mcmc(pos, 10000, progress=True)  # MCMC

                # discards gets rid of the first # iterations
                return sampler.get_chain(discard=100, thin=15, flat=True), sampler
            flat_samples, sampler = runner()

            # and thin chooses every 15

            def chainquality():  # Gives samples of each chain
                fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
                samples = sampler.get_chain()
                # Gives us what our MCMC variables around
                for i in range(len(labels)):
                    ax = axes[i]
                    ax.plot(samples[:, :, i],  alpha=0.4, color="red")
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                #plt.show()
                #plt.clf()
                axes[-1].set_xlabel("step number")
            #chainquality()

            def max_likelyhood():
                def error_from_MCMC(): # Finds the errors in the MCMC
                    errors = []
                    for i in range(len(labels)):
                        # 1 sigma from median
                        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                        q = np.diff(mcmc)  # gives sigma up and down
                        errors.append([mcmc[1], q[0], q[1]])
                    return errors

                def error_labelling(): # Labels the errors from the MCMC
                    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
                    for i in range(len(labels)):
                        txti = txt.format(
                            errors[i][0], errors[i][1], errors[i][2], labels[i])
                        # empty plots that display errors and values
                        plt.plot(0, 0, color="red", label=r"${}$".format(txti))

                #actual max likeyhood
                nll = lambda *args: -log_likelihood(*args) #empty initalize
                initial =  np.random.randn(len(labels)) #random values
                soln = minimize(nll, initial
                                ,method='SLSQP', bounds=((0,2),(-5,5),(0,100))#bounds nesssary as was getting weirdd answers 
                                ) #minimization
                maxlikelyhoodsolution = soln.x #values
                errors = error_from_MCMC()

                plt.plot(x0, x0*maxlikelyhoodsolution[0]+maxlikelyhoodsolution[1], alpha=1, color="red",
                         label=r"${}x+{}$".format(np.around(maxlikelyhoodsolution[0], 3), np.around(maxlikelyhoodsolution[1], 3)))
                error_labelling()  # adds labels for error in m,b

                # gives m,b in terms of likelyhood, median of MCMC, and the errors of MCMC, and variance and errors in MCMC
                return ([[maxlikelyhoodsolution[0], [errors[0][0], errors[0][1], errors[0][2]]], [maxlikelyhoodsolution[1], [errors[1][0], errors[1][1], errors[1][2]]], [maxlikelyhoodsolution[2], [errors[2][0], errors[2][1], errors[2][2]]]])
            
            values = max_likelyhood()

            inds = np.random.randint(len(flat_samples), size=100)

 
            for i in range(len(inds)): #plots the chosen values with mx+b
                yplot=[]
                sample = flat_samples[i]
                print(sample)
                yplot.append(np.add(x0*sample[0],sample[1])) #mx+b
                print(np.shape(yplot),np.shape(x0))
                plt.plot(np.reshape(x0,-1),np.reshape(yplot,-1), color="orange", alpha=0.1) 
                

            def correlation_time(): #Finds autocorrelation time and creates an empty plot for label
                tau = sampler.get_autocorr_time()
                plt.plot(0, 0, alpha=0,
                         label="Autocorrelation time m:"+str(np.around(tau[0], 3)))
                plt.plot(0, 0, alpha=0, label="Autocorrelation time b:" +
                         str(np.around(tau[1], 3)))
                plt.plot(0, 0, alpha=0, label="Autocorrelation time V:" +
                         str(np.around(tau[2], 3)))
            correlation_time()

            plt.errorbar(x=x[0], y=y[0], yerr=y[1],
                         xerr=x[1], fmt=".k", capsize=0,)

            plt.legend(fontsize=14)
            plt.title(np.around(redshift, 3))

            plt.xlabel("Log Flux UV offset: "+str(uvincrease))
            plt.ylabel("Log Flux X-Ray offset: "+str(xrayincrease))

            plt.savefig(str(np.around(redshift, 3))+".png", format='png')
            plt.clf()  # clears fig
            return values  # implemented
        values = MCMC()
    # gives array [[x,xerr],[y,yerr],gamma,betahat]
        return values
    values = Uncertainity_Calculations(array[0], array[1])
    return values


t = Table.read(r"Explore\table3local.fits")
t.add_column(0, name='subset')
t.sort('z')


# Define redshift subsets
# See https://docs.astropy.org/en/stable/table/modify_table.html

# Lengths of the redshift subsets (for a column sorted by increasing redshift)
subset_sizes = [77, 32, 37, 59, 83, 107, 149, 170, 186, 216,
                195, 241, 214, 215, 178, 110, 57, 48, 14]  # +33 left over

# Create array of starting array indices for the redshift subsets
subset_starts = [77, 32, 37, 59, 83, 107, 149, 170,
                 186, 216, 195, 241, 214, 215, 178, 110, 57, 48, 14]
subset_starts[0] = 0
for index in range(1, len(subset_sizes)):
    subset_starts[index] = subset_starts[index-1]+subset_sizes[index-1]

# print(subset_sizes)
# print(subset_starts)

# Set the 'subset' column in table t to an integer 1 to 18 for the 18 redshift subsets
# Ignore quasars in subset 0 (at both the lowest and highest redshifts)

zarray = []
MCMCslope = []
MCMCslopeerr = []
MCMCintercept = []
MCMCsloperrx = []
MCMCsloperry = []
MCMCintercepterrx = []
MCMCintercepterry = []
slopemax = []
interceptmax = []
zerror = []
MCMCvariance = []
variancemax = []
MCMCvarianceerrx = []
MCMCvarianceerry = []

uvincrease=28
xrayincrease=30

for index in range(0, len(subset_sizes)):  # ignore == 0
    starti = subset_starts[index] #start index
    endi = subset_starts[index]+subset_sizes[index] #end index
    t['subset'][starti:endi] = index #names the subset ie 0,1,2...
    x = [t['logFUV'][starti:endi]+uvincrease, t['e_logFUV'][starti:endi]]
    y = [t['logFX'][starti:endi]+xrayincrease, t['e_logFX'][starti:endi]]

    z = np.mean(t['z'][starti:endi]) #Finds the mean red-shift to be used in title
    #zerror.append([0,0]) #[np.abs(np.min(t['z'][starti:endi])),np.abs(np.min(t['z'][starti:endi]))]) #Founds the bounds (aka error of the z)
    zarray.append(z)
    values = (Main(z, [x, y])) 
    MCMCslope.append(values[0][1][0]) #ugly but numpy wasnt cooperating with me so this gives the slopes, intercepts, variance from maxlikelyhood and MCMC (with error values)
    MCMCsloperrx.append(values[0][1][1])
    MCMCsloperry.append(values[0][1][1])
    MCMCintercept.append(-1*values[1][1][0])
    MCMCintercepterrx.append(values[1][1][1])
    MCMCintercepterry.append(values[1][1][2])
    slopemax.append(values[0][0])
    interceptmax.append(-1*values[1][0])
    variancemax.append(values[2][0])
    MCMCvariance.append(values[2][1][0])
    MCMCvarianceerrx.append(values[2][1][1])
    MCMCvarianceerry.append(values[2][1][2])

def Plotting():
    plt.errorbar(x=zarray, y=MCMCslope,
                xerr=0, yerr=[MCMCsloperrx, MCMCsloperry], color="dodgerblue", label=" MCMC Slope Values") #errorbar of MCMC slope

    plt.scatter(x=zarray, y=MCMCslope, color="dodgerblue")
    plt.errorbar(x=zarray, y=MCMCintercept,
                xerr=0, yerr=[MCMCintercepterrx, MCMCintercepterry], color="red", label=" MCMC Intercept Values") 
    plt.scatter(x=zarray, y=MCMCintercept, color="red")
    plt.scatter(x=zarray, y=slopemax, label="Slope max like.", color="cyan")
    plt.scatter(x=zarray, y=interceptmax, label="Intercept max. Like", color="brown")

    plt.scatter(x=zarray, y=variancemax, label="max. like variance")
    plt.scatter(x=zarray, y=MCMCvariance, color="darkviolet")
    plt.errorbar(x=zarray, y=MCMCvariance, xerr=0, yerr=[
                MCMCvarianceerrx, MCMCintercepterry], color="darkviolet", label="MCMC variance")
    plt.ylabel("Parameters")
    plt.xlabel("Redshift")
    plt.title("Fit parameters over red-shift bins")
    plt.legend()
    plt.show()
Plotting()
