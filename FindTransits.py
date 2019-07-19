import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from bls import BLS
from lightkurve.lightcurve import LightCurve as LC
import exoplanet as xo
from wotan import flatten

import pymc3 as pm
import theano.tensor as tt
import corner


__all__ = ['FindTransits']

class FindTransits(object):
    '''Runs through the box-least-squares method to find transits and
    performs GP modeling'''

    def __init__(self, time, flux, flux_err):
        #super(, self).__init__()

        self.time = time
        self.flux = flux
        self.flux_err = flux_err

        self.find_rotper()
        #self.wotan_clan()
        self.do_bls()
        # self.make_bls_periodogram()
        # self.plot_box()
        # self.mask_transits()
	    # self.build_GPmodel()
	    # self.plot_light_curves()
        # self.plot_GPmodel()

    def find_rotper(self):
        '''
        '''
        results = xo.estimators.lomb_scargle_estimator(self.time, self.flux, max_peaks=1, min_period=0.1, max_period=30, samples_per_peak=50)

        peak = results['peaks'][0]
        freq, power = results['periodogram']
        # plt.plot(-np.log10(freq), power, "k")
        # plt.axvline(np.log10(peak["period"]), color="k", lw=4, alpha=0.3)
        # plt.xlim((-np.log10(freq)).min(), (-np.log10(freq)).max())
        # plt.yticks([])
        # plt.xlabel("log10(period)")
        # plt.ylabel("power")
        # plt.show()

        self.peak_rotper = peak["period"]

    def wotan_clan(self):
        '''GP model of stellar variability
        '''
        #self.flatten_lc, self.trend_lc = flatten(self.time, self.flux, method='gp', kernel='periodic_auto', kernel_size=5, return_trend=True, robust=True)
        self.flatten_lc, self.trend_lc = flatten(self.time, self.flux, method='rspline', window_length=0.1, break_tolerance=1, return_trend=True)

        plt.scatter(self.time, self.flux, s=1, color='black')
        plt.plot(self.time, self.trend_lc, color='green', linewidth=1, label='gp')
        plt.show()

        self.wotflux = self.flux/self.trend_lc - 1

    def do_bls(self):
        """
        """
        self.wotflux = self.flux
        durations = np.linspace(0.05, 0.2, 10)
        bls_model = BLS(self.time, self.wotflux)
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

        plt.show()
        print("The most likely period is" + " " +str(peak_period))

    def plot_box(self):
        """
        """
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.3)

        self.wotflux = self.flux
        # Plot the light curve and best-fit model
        ax = axes[0]
        ax.plot(self.time, self.wotflux, ".k", ms=3)
        x = np.linspace(self.time.min(), self.time.max(), 3*len(self.time))
        f = self.bls_model.model(x, self.bls_period, self.bls_duration, self.bls_t0)
        ax.plot(x, f, lw=0.75)
        ax.set_xlabel("time [days]")
        ax.set_ylabel("de-trended flux")

        # Plot the folded data points within 0.5 days of the transit time
        ax = axes[1]
        x = (self.time - self.bls_t0 + 0.5*self.bls_period) % self.bls_period - 0.5*self.bls_period
        m = np.abs(x) < 0.5
        ax.plot(x[m], self.wotflux[m], ".k", ms=3)

        # Over-plot the best fit model
        x = np.linspace(-0.5, 0.5, 1000)
        f = self.bls_model.model(x + self.bls_t0, self.bls_period, self.bls_duration, self.bls_t0)
        ax.plot(x, f, lw=0.75)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel("time since transit [days]")
        ax.set_ylabel("de-trended flux")

        plt.show()


    def mask_transits(self):
        """
        """
        start_times = []
        end_times = []

        for t in self.bls_model.compute_stats(self.bls_period, self.bls_duration, self.bls_t0)['transit_times']:
            start_times.append(t - (self.bls_duration/2))
            end_times.append(t + (self.bls_duration/2))

        corrected_start = np.array(start_times) - 0.02 #does 0.02 work?
        corrected_end = np.array(end_times) + 0.02

        mask_trns = np.ones(len(self.wotflux), dtype=bool)

        for i, t in enumerate(corrected_start):
            pre_mask = np.where((self.time > t) & (self.time < (corrected_end[i])))
            for j in pre_mask:
                mask_trns[j] = False

        self.mask_trns = mask_trns



    def build_GPmodel(self, mask=None, start=None):
        """from exoplanet"""

        if mask is None:
            mask = np.ones(len(self.time), dtype=bool)

        with pm.Model() as GPmodel:

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            # # Stellar parameters from Huang et al (2018)
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
            logrotperiod = pm.Normal("logrotperiod", mu=np.log(self.peak_rotper), sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # # Transit jitter & GP parameters
            # logs2 = pm.Normal("logs2", mu=np.log(np.var(self.flux[mask])), sd=10)
            # logw0_guess = np.log(2*np.pi/10)
            # logw0 = pm.Normal("logw0", mu=logw0_guess, sd=10)
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(self.flux_err[mask])), sd=5.0)

            # # We'll parameterize using the maximum power (S_0 * w_0^4) instead of
            # # S_0 directly because this removes some of the degeneracies between
            # # S_0 and omega_0
            # logpower = pm.Normal("logpower", mu=np.log(np.var(self.flux[mask]))+4*logw0_guess, sd=10)
            # logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))
            # Track the rotation period as a deterministic
            rotperiod = pm.Deterministic("rotation_period", tt.exp(logrotperiod))


            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=t0, b=b, ecc=ecc, omega=omega)

            # Compute the model light curve using starry, r = r_pl
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl, t=self.time[mask], texp=0.021)
            light_curve = pm.math.sum(light_curves, axis=-1) #+ mean
            pm.Deterministic("light_curves", light_curves)


            # GP model for the light curve
            # kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
            # gp = xo.gp.GP(kernel, self.time[mask], tt.exp(logs2) + tt.zeros(mask.sum()), J=2)
            kernel = xo.gp.terms.RotationTerm(log_amp=logamp, period=rotperiod, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix)
            gp = xo.gp.GP(kernel, self.time[mask], ((self.flux_err[mask])**2 + tt.exp(logs2)), J=4)

            # pm.Potential("transit_obs", gp.log_likelihood(self.flux[mask] - light_curve))
            # pm.Deterministic("gp_pred", gp.predict())
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
            # map_soln = xo.optimize(start=start, vars=[logs2, logpower, logw0])
            map_soln = xo.optimize(start=start, vars=[mean])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
            map_soln = xo.optimize(start=map_soln, vars=[u_star])
            map_soln = xo.optimize(start=map_soln, vars=[logr])
            map_soln = xo.optimize(start=map_soln, vars=[b])
            map_soln = xo.optimize(start=map_soln, vars=[ecc, omega])
            map_soln = xo.optimize(start=map_soln, vars=[mean])
            # map_soln = xo.optimize(start=map_soln, vars=[logs2, logpower, logw0])
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

            # ############# Stellar variability model ############################
            # # A jitter term describing excess white noise
            # logs2 = pm.Normal("logs2", mu=2*np.log(np.min(self.flux_err)), sd=5.0)
            #
            # # The parameters of the RotationTerm kernel
            # logamp = pm.Normal("logamp", mu=np.log(np.var(self.flux)), sd=5.0)
            # logrotperiod = pm.Normal("logperiod", mu=np.log(self.peak_rotper), sd=5.0)
            # logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            # logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            # mix = pm.Uniform("mix", lower=0, upper=1.0)
            #
            # # Track the rotation period as a deterministic
            # rotperiod = pm.Deterministic("rotation_period", tt.exp(logrotperiod))
            #
            # # Set up the Gaussian Process model
            # kernel = xo.gp.terms.RotationTerm(log_amp=logamp, period=logrotperiod, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix)
            # gp = xo.gp.GP(kernel, self.time, ((self.flux_err)**2 + tt.exp(logs2)), J=4)
            #
            # # Compute the Gaussian Process likelihood and add it into the
            # # the PyMC3 model as a "potential"
            # pm.Potential("loglike", gp.log_likelihood((self.flux) - mean))
            #
            # # Compute the mean model prediction for plotting purposes
            # pm.Deterministic("pred", gp.predict())
            #
            # # Optimize to find the maximum a posteriori parameters
            # map_soln = xo.optimize(start=GPmodel.test_point)

        ########################################################################

    def build_no_pl_GPmodel(self, mask=None, start=None):
        """from exoplanet"""

        if mask is None:
            mask = np.ones(len(self.time), dtype=bool)

        with pm.Model() as no_pl_GPmodel:

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            # # Stellar parameters from Huang et al (2018)
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
            logrotperiod = pm.Normal("logrotperiod", mu=np.log(self.peak_rotper), sd=5.0)
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
            light_curve = pm.math.sum(light_curves, axis=-1) #+ mean
            pm.Deterministic("light_curves", light_curves)


            # GP model for the light curve

            kernel = xo.gp.terms.RotationTerm(log_amp=logamp, period=rotperiod, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix)
            gp = xo.gp.GP(kernel, self.time[mask], ((self.flux_err[mask])**2 + tt.exp(logs2)), J=4)

            # pm.Potential("transit_obs", gp.log_likelihood(self.flux[mask] - light_curve))
            # pm.Deterministic("gp_pred", gp.predict())
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



    def plot_lc(self, soln, mask=None):
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

        plt.show()


    def plot_GPmodel(self, GPmodel, soln, mask=None):
        '''
        '''
        with GPmodel:
            mu = xo.eval_in_model(self.gp.predict(self.time, return_var=False), soln)

        plt.plot(self.time, self.flux, "k", lw=1.5, alpha=0.3, label="truth")

        plt.plot(self.time, mu+1, color="C1", label="prediction")

        gp_mod = soln["pred"] + soln["mean"]
        fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharex=True)
        ax = axes
        ax.plot(np.mod(self.time, soln["period"]), self.flux - mu, "k.", label="de-trended data")

        for i, l in enumerate("b"):
            mod = soln["light_curves"][:, i]
            ax.plot(np.mod(self.time[mask], soln["period"]), mod+1, "b.", label="model")

        ax.legend(fontsize=10, loc=3)
        ax.set_ylabel("de-trended flux")

        plt.show()

    # def sample_model(self, model, tune=500):
    #     '''
    #     '''
    #     np.random.seed(42)
    #     sampler = xo.PyMC3Sampler(finish=300, chains=4)
    #     with GPmodel:
    #         burnin = sampler.tune(tune=tune, start=map_soln, step_kwargs=dict(target_accept=0.9))
    #         trace = sampler.sample(draws=2000)
    #
    #     self.trace = trace
    #     pm.summary(trace, varnames=["logw0", "logpower", "logs2", "omega", "ecc", "r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean"])



    def plot_folded_lc(self, trace, mask):
        '''
        '''
        # Compute the GP prediction
        gp_mod = np.median(trace["pred"] + trace["mean"][:, None], axis=0)

        # Get the posterior median orbital parameters
        p = np.median(trace["period"])
        t0 = np.median(trace["t0"])

        # Plot the folded data
        x_fold = (self.time[mask] - t0 + 0.5*p) % p - 0.5*p
        plt.plot(x_fold, self.flux[mask] - gp_mod, ".k", label="data", zorder=-1000)

        # Overplot the phase binned light curve
        bins = np.linspace(-0.41, 0.41, 50)
        denom, _ = np.histogram(x_fold, bins)
        num, _ = np.histogram(x_fold, bins, weights=self.flux[mask])
        denom[num == 0] = 1.0
        plt.plot(0.5*(bins[1:] + bins[:-1]), num / denom, "o", color="C2",
                 label="binned")

        # Plot the folded model
        inds = np.argsort(x_fold)
        inds = inds[np.abs(x_fold)[inds] < 0.3]
        pred = trace["light_curves"][:, inds, 0]
        pred = np.percentile(pred, [16, 50, 84], axis=0)
        plt.plot(x_fold[inds], pred[1], color="C1", label="model")
        art = plt.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5,
                               zorder=1000)
        art.set_edgecolor("none")

        # Annotate the plot with the planet's period
        txt = "period = {0:.5f} +/- {1:.5f} d".format(
            np.mean(trace["period"]), np.std(trace["period"]))
        plt.annotate(txt, (0, 0), xycoords="axes fraction",
                     xytext=(5, 5), textcoords="offset points",
                     ha="left", va="bottom", fontsize=12)

        plt.legend(fontsize=10, loc=4)
        plt.xlim(-0.5*p, 0.5*p)
        plt.xlabel("time since transit [days]")
        plt.ylabel("de-trended flux")
        plt.xlim(-0.15, 0.15);


    def plot_corner(self, trace):
        '''
        '''
        varnames = ["period", "b", "ecc", "r_pl"]
        samples = pm.trace_to_dataframe(trace, varnames=varnames)

        # Convert the radius to Earth radii
        samples["r_pl"] = (np.array(samples["r_pl"]) * u.R_sun).to(u.R_earth).value

        corner.corner(samples[["period", "r_pl", "b", "ecc"]], labels=["period [days]", "radius [Earth radii]", "impact param", "eccentricity"]);
