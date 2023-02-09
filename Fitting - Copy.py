#Cassandra
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import emcee
from scipy.optimize import minimize
import corner
t = Table.read(r"Explore\table3local.fits") #import file, note may have to change import path as for me it was locally Explore/<>
delta=[[1,2],[3,4],[6,7]]
print(np.array(np.reshape(delta,-1))**2)
def Main():

    # Define line parameters (for testing) 
    gamma=0.6
    betahat=0 # may need to change

    def Uncertainity_Calculations(x,y,gamma,betahat):
        x0=np.linspace(-1.5,3,300)
        # Define matrix Z with shape 2,2421 : each column is a separate column vector [x_i, y_i]

        # Define the unit vector orthogonal to the defined line: Hogg+ Eq. 29

        z=np.matrix([x[0], y[0]])

        
        def MCMC():
            def Orthogonal_Displacements(m,b): 
                # Define matrix Delta (Hogg+ Eq. 30) with shape 1,2421 consisting of the
                #  orthogonal displacements of each data point x_i,y_i from the defined line
                vee = (np.matrix([-m, 1]).T / np.sqrt(1+m**2.))
                Delta = vee.T * z - b / np.sqrt(1+m**2.)
                return np.square(Delta)
            def Orthogonal_Variance(m):
                            zero = np.zeros_like(x[1])#returns an array with the same shape as the x_errors
                            part1 = (np.array([x[1],zero,zero,y[1]]).T).reshape(len(x[0]),2,2) #Creates the 2x2 covariance matrix
                            vee = (np.matrix([-m, 1]).T / np.sqrt(1+m**2.))
                            Sigma1 = np.matmul(part1,vee) #Covariance *v
                            #print(Sigma1)
                            Sigma2 = np.matmul(Sigma1,vee) #"squares" our covariance
                            return Sigma2
            def log_likelihood(theta):
                m, b,v= theta
                deltasquared=Orthogonal_Displacements(m,b)
                sigmasquared=Orthogonal_Variance(m)
                LnlVariance=-np.sum(1/2*np.log(sigmasquared+v))-np.sum((deltasquared/(2*(sigmasquared+v))))
                return LnlVariance



            def log_prior(theta):
                m, b,v = theta
                if 0 < m < 2 and -7 < b < 7 and 0<v<1000:
                    return 0.0
                return -np.inf

            def log_probability(theta):
                lp = log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                blah=log_likelihood(theta)
                return lp + blah



            pos=np.random.rand(30,3)
            nwalkers, ndim = pos.shape

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability,
            )
            sampler.run_mcmc(pos, 30000,progress=True)

            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)


            fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
            samples = sampler.get_chain()
            labels = ["m", "b", "V"]
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            plt.show()
            plt.clf()
            axes[-1].set_xlabel("step number")
            inds = np.random.randint(len(flat_samples), size=100)
            for ind in inds:
                sample = flat_samples[ind]
                plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
            plt.errorbar(x=x[0], y=y[0], yerr=y[1],xerr=x[1], fmt=".k", capsize=0)
            plt.legend(fontsize=14)
            plt.show()
            tau = sampler.get_autocorr_time()
            print(tau)
        MCMC()
    Uncertainity_Calculations([t['logFUV']+27.5,t['e_logFUV']],[t['logFX']+31.5,t['e_logFX']],gamma,betahat) #gives array [[x,xerr],[y,yerr],gamma,betahat]
Main()