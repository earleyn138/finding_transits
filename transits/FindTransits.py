import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from operator import itemgetter

from bls import BLS
import exoplanet as xo

import pymc3 as pm
import theano.tensor as tt
import corner



__all__ = ['FindTransits']

class FindTransits(object):
    '''Runs through the box-least-squares method to find transits and
    performs GP modeling'''

    def __init__(self, time, flux, flux_err, cads, tic, run):

        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.cads = cads
        self.tic = tic
        self.run = run

        self.find_rotper(time, flux)
        self.make_lombscarg(time, flux)
        self.fft_lc()
        self.do_bls()


    def find_rotper(self, time, flux):
        '''
        '''
        ls_results = xo.estimators.lomb_scargle_estimator(time, flux, max_peaks=1, min_period=0.1, max_period=30, samples_per_peak=50)
        peak = ls_results['peaks'][0]

        return peak['period'], ls_results


    def make_lombscarg(self, time, flux):
        '''
        '''
        rotper, ls_results = self.find_rotper(time, flux)
        freq, power = ls_results['periodogram']

        plt.plot(-np.log10(freq), power, "k")
        plt.axvline(np.log10(rotper), color="k", lw=4, alpha=0.3)
        plt.xlim((-np.log10(freq)).min(), (-np.log10(freq)).max())
        plt.yticks([])
        plt.ylabel("power")

        if flux is self.flux:
            plt.title('The rotation period from Lomb-Scargle is {}'.format(rotper))
            plt.xlabel('log10(period)')
            plt.savefig(fname='/home/earleyn/figures/raw_lombscarg_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
            #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/raw_lombscarg_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')

        elif flux is self.det_flux:
            plt.title('The rotation period after detrending is {}'.format(rotper))
            plt.xlabel('log10(period)')
            plt.savefig(fname='/home/earleyn/figures/det_lombscarg_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
            #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/det_lombscarg_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')

        plt.close()



    def fft_lc(self):

        # Creating a new array of flux that are equally spaced apart --accounts for missing cads
        even_flux = np.array(int(self.cads.max()-self.cads.min()+1)*[1.0000001])

        for i, cad in enumerate(self.cads):
            index = int(cad - self.cads.min())
            even_flux[index] = self.flux[i]

        # Finding rotation period on raw data
        rotper, ls_results = self.find_rotper(self.time, self.flux)

        self.make_lombscarg(self.time, self.flux)

        #Fast fourier transform
        fft_flux = np.fft.fft(even_flux)
        freq = np.fft.fftfreq(even_flux.shape[-1], d=(1/48))
        period = 1/freq

        fft_power1 = np.sqrt(fft_flux.real**2+fft_flux.imag**2)

        power_tup = []
        for i, per in enumerate(period):
            if per < rotper+1 and per > rotper-1:
                power_tup.append((i, per))

        fft_list = []
        for tup in power_tup:
            fft_list.append((fft_power1[tup[0]], tup[1]))

        max_power = max(fft_list,key=itemgetter(0))[0]
        max_period = max(fft_list,key=itemgetter(0))[1]

        plt.plot(period, fft_power1)
        plt.axvline(max_period, color="r", lw=4, alpha=0.3)
        plt.axvline(-max_period, color="r", lw=4, alpha=0.3)
        plt.xlim(-1.1*max_period, 1.1*max_period)
        plt.savefig(fname='/home/earleyn/figures/fft_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/fft_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        plt.close()

        #Signal processing: Top hat filter
        #Bounds
        pos_low_bound = 0.75*max_period
        pos_up_bound = 1.25*max_period
        neg_low_bound = -1.25*max_period
        neg_up_bound = -0.75*max_period
        fft_power_cut = 1.

        for i, per in enumerate(period):
            if per > pos_low_bound and per < pos_up_bound:
                fft_flux[i] = 0

            elif per > neg_low_bound and per < neg_up_bound:
                fft_flux[i] = 0

            elif (per < pos_low_bound and per > 0) and (fft_power1[i] > fft_power_cut):
                if i-2 >= 0 and i+3 <= len(fft_flux)-1:
                    for j in range(i-2, i+3):
                        fft_flux[j] = 0

            elif (per > pos_up_bound and per < 10) and (fft_power1[i] > fft_power_cut):
                if i-2 >= 0 and i+3 <= len(fft_flux)-1:
                    for j in range(i-2, i+3):
                        fft_flux[j] = 0

            elif (per < neg_low_bound and per > -10) and (fft_power1[i] > fft_power_cut):
                if i-2 >= 0 and i+3 <= len(fft_flux)-1:
                    for j in range(i-2, i+3):
                        fft_flux[j] = 0

            elif (per > neg_up_bound and per < 0) and (fft_power1[i] > fft_power_cut):
                if i-2 >= 0 and i+3 <= len(fft_flux)-1:
                    for j in range(i-2, i+3):
                        fft_flux[j] = 0

        fft_power2 = np.sqrt(fft_flux.real**2+fft_flux.imag**2)


        plt.plot(period, fft_power2)
        plt.axvline(max_period, color="r", lw=4, alpha=0.3)
        plt.axvline(-max_period, color="r", lw=4, alpha=0.3)
        plt.xlim(neg_low_bound-0.1*max_period, pos_up_bound+0.1*max_period)
        plt.savefig(fname='/home/earleyn/figures/notch_filter_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/notch_filter_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        plt.close()

        #Inverse fourier transform
        ifft_flux = np.fft.ifft(fft_flux)

        pflux = np.sqrt(ifft_flux.real**2+ifft_flux.imag**2) #processed flux

        plt.plot(pflux)
        plt.xlabel('Cadences')
        plt.ylabel('Detrended Normalized Flux')
        plt.savefig(fname='/home/earleyn/figures/det_lc_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/det_lc_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        plt.close()

        det_flux = []
        for value in (np.where(even_flux != 1.0000001))[0]:
            det_flux.append(pflux[value])

        det_flux = np.asarray(det_flux) #Final detrended flux

        self.det_flux = det_flux

        # Confirming to see that rotation period is not preserved in processed data
        rotper, ls_results = self.find_rotper(self.time, self.det_flux)

        self.make_lombscarg(self.time, self.det_flux)


    def do_bls(self):
        """
        """
        self.bls_time = self.time[20:981]
        self.bls_flux = self.det_flux[20:981]

        durations = np.linspace(0.05, 0.2, 10)
        bls_model = BLS(self.bls_time, self.bls_flux)
        bls_results = bls_model.autopower(durations, frequency_factor=5.0)
        self.bls_results = bls_results

        index = np.argmax(bls_results.power)
        bls_period = bls_results.period[index]
        bls_t0 = bls_results.transit_time[index]
        bls_duration = bls_results.duration[index]
        bls_depth = bls_results.depth[index]

        self.bls_model = bls_model
        self.bls_period = bls_period
        self.bls_t0 = bls_t0
        self.bls_depth = bls_depth
        self.bls_duration = bls_duration


    def make_bls_periodogram(self):
        """
        """
        # Find the period of the peak
        peak_period = self.bls_results.period[np.argmax(self.bls_results.power)]

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))

        # Highlight the harmonics of the peak period
        ax.axvline(peak_period, alpha=0.4, lw=3)
        for n in range(2, 10):
            ax.axvline(n*peak_period, alpha=0.4, lw=1, linestyle="dashed")
            ax.axvline(peak_period/n, alpha=0.4, lw=1, linestyle="dashed")

        # Plot the periodogram
        ax.plot(self.bls_results.period, self.bls_results.power, "k", lw=0.5)

        ax.set_xlim(self.bls_results.period.min(), self.bls_results.period.max())
        ax.set_xlabel("period [days]")
        ax.set_ylabel("log likelihood")

        plt.savefig(fname='/home/earleyn/figures/bls_pgram_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/bls_pgram_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        plt.close()


    def plot_box(self):
        """
        """
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.3)

        # Plot the light curve and best-fit model
        ax = axes[0]
        ax.plot(self.bls_time, self.bls_flux, ".k", ms=3)
        x = np.linspace(self.bls_time.min(), self.bls_time.max(), 3*len(self.bls_time))
        f = self.bls_model.model(x, self.bls_period, self.bls_duration, self.bls_t0)
        ax.plot(x, f, lw=0.75)
        ax.set_xlabel("time [days]")
        ax.set_ylabel("de-trended flux")

        # Plot the folded data points within 0.5 days of the transit time
        ax = axes[1]
        x = (self.bls_time - self.bls_t0 + 0.5*self.bls_period) % self.bls_period - 0.5*self.bls_period
        m = np.abs(x) < 0.5
        ax.plot(x[m], self.bls_flux[m], ".k", ms=3)

        # Over-plot the best fit model
        x = np.linspace(-0.5, 0.5, 1000)
        f = self.bls_model.model(x + self.bls_t0, self.bls_period, self.bls_duration, self.bls_t0)
        ax.plot(x, f, lw=0.75)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel("time since transit [days]")
        ax.set_ylabel("de-trended flux")

        plt.savefig(fname='/home/earleyn/figures/box_plot_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/box_plot_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
        plt.close()



    def build_GPmodel(self, mask=None, start=None):
        """from exoplanet"""

        # Find rotation period
        rotper, ls_results = self.find_rotper(self.time, self.flux)

        if mask is None:
            mask = np.ones(len(self.time), dtype=bool)

        with pm.Model() as GPmodel:

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            # Stellar parameters from Huang et al (2018)
            M_star_huang = 1.094, 0.039
            R_star_huang = 1.10, 0.023
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
            m_star = BoundedNormal("m_star", mu=M_star_huang[0], sd=M_star_huang[1])
            r_star = BoundedNormal("r_star", mu=R_star_huang[0], sd=R_star_huang[1])

            # Orbital parameters for the planets
            logP = pm.Normal("logP", mu=np.log(self.bls_period), sd=1)
            t0 = pm.Normal("t0", mu=self.bls_t0, sd=1)
            b = pm.Flat("b", transform=pm.distributions.transforms.logodds, testval=0.5)
            logr = pm.Normal("logr", sd=1.0, mu=0.5*np.log(np.array(self.bls_depth))+np.log(R_star_huang[0]))
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            ror = pm.Deterministic("ror", r_pl / r_star)

            # This is the eccentricity prior from Kipping (2013):
            # https://arxiv.org/abs/1306.4982
            BoundedBeta = pm.Bound(pm.Beta, lower=0, upper=1-1e-5)
            ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, testval=0.1)
            omega = xo.distributions.Angle("omega")

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.flux[mask])), sd=5.0)
            logrotperiod = pm.Normal("logrotperiod", mu=np.log(rotper), sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Transit jitter & GP parameters
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(self.flux_err[mask])), sd=5.0)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))

            # Track the rotation period as a deterministic
            rotperiod = pm.Deterministic("rotation_period", tt.exp(logrotperiod))

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=t0, b=b, ecc=ecc, omega=omega)

            # Compute the model light curve using starry, r = r_pl
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl, t=self.time[mask], texp=0.021)
            light_curve = pm.math.sum(light_curves, axis=-1)
            pm.Deterministic("light_curves", light_curves)

            # GP model for the light curve
            kernel = xo.gp.terms.RotationTerm(log_amp=logamp, period=rotperiod, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix)
            gp = xo.gp.GP(kernel, self.time[mask], ((self.flux_err[mask])**2 + tt.exp(logs2)), J=4)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(self.flux[mask] - mean - light_curve))

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())
            pm.Deterministic("loglikelihood", gp.log_likelihood(self.flux[mask] - mean - light_curve))


            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = GPmodel.test_point
            map_soln = xo.optimize(start=start, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
            map_soln = xo.optimize(start=map_soln, vars=[u_star])
            map_soln = xo.optimize(start=map_soln, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[ecc, omega])
            map_soln = xo.optimize(start=map_soln, vars=[mean])
            map_soln = xo.optimize(start=map_soln)

            # Optimize to find the maximum a posteriori parameters
            map_soln = xo.optimize(start=map_soln, vars=[logs2, logQ0, logdeltaQ])
            map_soln = xo.optimize(start=map_soln, vars=[logamp])
            map_soln = xo.optimize(start=map_soln, vars=[logrotperiod])
            map_soln = xo.optimize(start=map_soln, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[mix])
            map_soln = xo.optimize(start=map_soln, vars=[logs2, logQ0, logdeltaQ])
            map_soln = xo.optimize(start=map_soln)

            self.gp = gp

        return GPmodel, map_soln


    def build_no_pl_GPmodel(self, mask=None, start=None):
        """from exoplanet"""

        #Find rotation period
        rotper, ls_results = self.find_rotper(self.time, self.flux)

        if mask is None:
            mask = np.ones(len(self.time), dtype=bool)

        with pm.Model() as no_pl_GPmodel:

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            # Stellar parameters from Huang et al (2018)
            M_star_huang = 1.094, 0.039
            R_star_huang = 1.10, 0.023
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
            m_star = BoundedNormal("m_star", mu=M_star_huang[0], sd=M_star_huang[1])
            r_star = BoundedNormal("r_star", mu=R_star_huang[0], sd=R_star_huang[1])

            # Orbital parameters for the planets
            logP = pm.Normal("logP", mu=np.log(self.bls_period), sd=1)
            t0 = pm.Normal("t0", mu=self.bls_t0, sd=1)
            b = pm.Flat("b", transform=pm.distributions.transforms.logodds, testval=0.5)
            logr = pm.Normal("logr", sd=1.0, mu=0.5*np.log(np.array(self.bls_depth))+np.log(R_star_huang[0]))
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            ror = pm.Deterministic("ror", r_pl / r_star)

            # This is the eccentricity prior from Kipping (2013):
            # https://arxiv.org/abs/1306.4982
            BoundedBeta = pm.Bound(pm.Beta, lower=0, upper=1-1e-5)
            ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, testval=0.1)
            omega = xo.distributions.Angle("omega")

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(self.flux[mask])), sd=5.0)
            logrotperiod = pm.Normal("logrotperiod", mu=np.log(rotper), sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Transit jitter & GP parameters
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(self.flux_err[mask])), sd=5.0)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))
            # Track the rotation period as a deterministic
            rotperiod = pm.Deterministic("rotation_period", tt.exp(logrotperiod))

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=t0, b=b, ecc=ecc, omega=omega)

            # Compute the model light curve using starry, r = 0.0 --> no planet!
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit, r=0.0, t=self.time[mask], texp=0.021)
            light_curve = pm.math.sum(light_curves, axis=-1)
            pm.Deterministic("light_curves", light_curves)


            # GP model for the light curve
            kernel = xo.gp.terms.RotationTerm(log_amp=logamp, period=rotperiod, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix)
            gp = xo.gp.GP(kernel, self.time[mask], ((self.flux_err[mask])**2 + tt.exp(logs2)), J=4)


            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(self.flux[mask] - mean - light_curve))

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())
            pm.Deterministic("loglikelihood", gp.log_likelihood(self.flux[mask] - mean - light_curve))


            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = no_pl_GPmodel.test_point
            no_pl_map_soln = xo.optimize(start=start, vars=[mean])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[b])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[logP, t0])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[u_star])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[logr])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[b])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[ecc, omega])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[mean])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln)
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[logs2, logQ0, logdeltaQ])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[logamp])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[logrotperiod])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[mean])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[mix])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln, vars=[logs2, logQ0, logdeltaQ])
            no_pl_map_soln = xo.optimize(start=no_pl_map_soln)

        return no_pl_GPmodel, no_pl_map_soln



    def plot_lc(self, soln, mask=None, pl=True):
        '''
		'''
        if mask is None:
            mask = np.ones(len(self.time), dtype=bool)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        ax = axes[0]
        ax.plot(self.time[mask], self.flux[mask], "k", label="data")
        # gp_mod = soln["gp_pred"] + soln["mean"]
        gp_mod = soln["pred"] + soln["mean"]
        ax.plot(self.time[mask], gp_mod, color="C2", label="gp model")
        ax.legend(fontsize=10)
        ax.set_ylabel("relative flux")

        ax = axes[1]
        ax.plot(self.time[mask], self.flux[mask] - gp_mod, "k", label="de-trended data")
        for i, l in enumerate("b"):
            mod = soln["light_curves"][:, i]
            ax.plot(self.time[mask], mod, label="planet {0}".format(l))
        ax.legend(fontsize=10, loc=3)
        ax.set_ylabel("de-trended flux")

        ax = axes[2]
        mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
        ax.plot(self.time[mask], self.flux[mask] - mod, "k")
        ax.axhline(0, color="#aaaaaa", lw=1)
        ax.set_ylabel("residuals")
        ax.set_xlim(self.time[mask].min(), self.time[mask].max())
        ax.set_xlabel("time [days]")

        if pl is True:
            plt.savefig(fname='/home/earleyn/figures/GPmodel_lc_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
            #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/GPmodel_lc_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')

        else:
            plt.savefig(fname='/home/earleyn/figures/no_pl_GPmodel_lc_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')
            #plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/no_pl_GPmodel_lc_tic{:d}_run{:d}'.format(self.tic, self.run), dpi=250, format='pdf')

        plt.close()
