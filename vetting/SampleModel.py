import theano.tensor as tt
import pymc3 as pm
import numpy as np
import exoplanet as xo
import corner
import astropy.units as u
import matplotlib.pyplot as plt

__all__ = ['SampleModel']

class SampleModel(object):
    ''' sampling'''

    def __init__(self, time, flux, flux_err, mask, soln, do_even_odd):

        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.mask = mask
        self.soln = soln
        self.do_even_odd = do_even_odd

        self.recreate_mod()
        self.sample_mod()


    def recreate_mod(self):
        '''
        '''
        with pm.Model() as self.model:

            # Parameters for the stellar properties
            mean = self.soln['mean']
            u_star = self.soln['u_star']

            if self.do_even_odd == False:
                logP = self.soln['logP']
                t0 = self.soln['t0']
                period = self.soln['period']
                m_star = self.soln['m_star']
                r_star = self.soln['r_star']
                b = self.soln['b']
                logr = self.soln['logr']
                r_pl = self.soln['r_pl']
                ror = self.soln['ror']
                ecc = self.soln['ecc']
                omega = self.soln['omega']

            # Even-Odd Test
            else:
                logP_even = self.soln['logP_even']
                t0_even = self.soln['t0_even']
                period_even = self.soln['period_even']
                m_star_even = self.soln['m_star_even']
                r_star_even = self.soln['r_star_even']
                b_even = self.soln['b_even']
                logr_even = self.soln['logr_even']
                r_pl_even = self.soln['r_pl_even']
                ror_even = self.soln['ror_even']
                ecc_even = self.soln['ecc_even']
                omega_even = self.soln['omega_even']

                logP_odd = self.soln['logP_odd']
                t0_odd = self.soln['t0_odd']
                period_odd = self.soln['period_odd']
                m_star_odd = self.soln['m_star_odd']
                r_star_odd = self.soln['r_star_odd']
                b_odd = self.soln['b_odd']
                logr_odd = self.soln['logr_odd']
                r_pl_odd = self.soln['r_pl_odd']
                ror_odd = self.soln['ror_odd']
                ecc_odd = self.soln['ecc_odd']
                omega_odd = self.soln['omega_odd']


            # The parameters of the RotationTerm kernel
            logamp = self.soln['logamp']
            logrotperiod = self.soln['logrotperiod']
            rotperiod=np.exp(logrotperiod)
            logQ0 = self.soln['logQ0']
            logdeltaQ = self.soln['logdeltaQ']
            mix = self.soln['mix']

            # Transit jitter & GP parameters
            logs2 = self.soln['logs2']

            # GP model for the light curve
            kernel = xo.gp.terms.RotationTerm(log_amp=logamp, period=rotperiod, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix)
            gp = xo.gp.GP(kernel, self.time[self.mask], ((self.flux_err[self.mask])**2 + tt.exp(logs2)), J=4)


            if self.do_even_odd == False:
                # Orbit model
                orbit = xo.orbits.KeplerianOrbit(r_star=r_star, m_star=m_star, period=period, t0=t0, b=b, ecc=ecc, omega=omega)
                light_curves = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl, t=self.time[self.mask], texp=0.021)

                light_curve = pm.math.sum(light_curves, axis=-1)
                pm.Deterministic("light_curves", light_curves)

                # Compute the Gaussian Process likelihood and add it into the
                # the PyMC3 model as a "potential"
                pm.Potential("loglike", gp.log_likelihood(self.flux[self.mask] - mean - light_curve))

                # Compute the mean model prediction for plotting purposes
                pm.Deterministic("pred", gp.predict())
                pm.Deterministic("loglikelihood", gp.log_likelihood(self.flux[self.mask] - mean - light_curve))


            else:
                orbit_even = xo.orbits.KeplerianOrbit(r_star=r_star_even, m_star=m_star_even, period=period_even, t0=t0_even, b=b_even, ecc=ecc_even, omega=omega_even)

                orbit_odd = xo.orbits.KeplerianOrbit(r_star=r_star_odd, m_star=m_star_odd, period=period_odd, t0=t0_odd, b=b_odd, ecc=ecc_odd, omega=omega_odd)

                light_curves_even = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit_even, r=r_pl_even, t=self.time[self.mask], texp=0.021)
                light_curves_odd = xo.StarryLightCurve(u_star).get_light_curve(orbit=orbit_odd, r=r_pl_odd, t=self.time[self.mask], texp=0.021)

                light_curve_even = pm.math.sum(light_curves_even, axis=-1)
                light_curve_odd = pm.math.sum(light_curves_odd, axis=-1)

                pm.Deterministic("light_curves_even", light_curves_even)
                pm.Deterministic("light_curves_odd", light_curves_odd)

                # Compute the Gaussian Process likelihood and add it into the
                # the PyMC3 model as a "potential"
                pm.Potential("loglike", gp.log_likelihood(self.flux[self.mask] - mean - (light_curve_even + light_curve_odd)))

                # Compute the mean model prediction for plotting purposes
                pm.Deterministic("pred", gp.predict())
                pm.Deterministic("loglikelihood", gp.log_likelihood(self.flux[self.mask] - mean - (light_curve_even + light_curve_odd)))



    def sample_mod(self):
        '''
        '''
        # Sampling model
        np.random.seed(42)
        sampler = xo.PyMC3Sampler(finish=300, chains=4)

        with self.model:
            burnin = sampler.tune(tune=800, start=self.soln, step_kwargs=dict(target_accept=0.9))
            self.trace = sampler.sample(draws=2000)


    # def assess_conv(self, do_even_odd):
    #     '''
    #     '''
    #     if do_even_odd == False:
    #         pm.summary(self.trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "omega", "ecc", "r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean", "logrotperiod"])
    #     else:
    #         pm.summary(self.trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "mean", "u_star", "logrotperiod", "omega_even", "ecc_even", "r_pl_even", "b_even", "t0_even", "logP_even", "r_star_even", "m_star_even", "omega_odd", "ecc_odd", "r_pl_odd", "b_odd", "t0_odd", "logP_odd", "r_star_odd", "m_star_odd" ])


    def plot_corner(self, varnames):
        '''
        '''
        #varnames = ["period", "b", "ecc", "r_pl"]
        samples = pm.trace_to_dataframe(self.trace, varnames=varnames)

        # Convert the radius to Earth radii
        samples["r_pl"] = (np.array(samples["r_pl"]) * u.R_sun).to(u.R_earth).value

        corner.corner(samples[["period", "r_pl", "b", "ecc"]], labels=["period [days]", "radius [Earth radii]", "impact param", "eccentricity"])
        plt.show()
        plt.close()






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
    #
    #
    #
    # def plot_corner(self, trace):
    #     '''
    #     '''
    #     varnames = ["period", "b", "ecc", "r_pl"]
    #     samples = pm.trace_to_dataframe(trace, varnames=varnames)
    #
    #     # Convert the radius to Earth radii
    #     samples["r_pl"] = (np.array(samples["r_pl"]) * u.R_sun).to(u.R_earth).value
    #
    #     corner.corner(samples[["period", "r_pl", "b", "ecc"]], labels=["period [days]", "radius [Earth radii]", "impact param", "eccentricity"])
