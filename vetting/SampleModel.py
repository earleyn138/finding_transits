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
            mean = pm.Normal("mean", mu=self.soln['mean'], sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")
            # Stellar parameters from Huang et al (2018)
            M_star_huang = 1.094, 0.039
            R_star_huang = 1.10, 0.023
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)

            if self.do_even_odd == False:
                logP = pm.Normal("logP", mu=self.soln['logP'], sd=1)
                t0 = pm.Normal("t0", mu=self.soln['t0'], sd=1)
                period = pm.Deterministic("period", tt.exp(logP))
                m_star = BoundedNormal("m_star", mu=self.soln['m_star'], sd=M_star_huang[1])
                r_star = BoundedNormal("r_star", mu=self.soln['r_star'], sd=R_star_huang[1])
                b = pm.Uniform("b", lower=0, upper=0.9, testval=self.soln['b'])
                BoundedNormal_logr = pm.Bound(pm.Normal, lower=-5, upper=0)
                logr = BoundedNormal_logr('logr', mu=self.soln['logr'], sd=1.0)
                r_pl = pm.Deterministic("r_pl", tt.exp(logr))
                ror = pm.Deterministic("ror", r_pl / r_star)
                BoundedBeta = pm.Bound(pm.Beta, lower=0, upper=1-1e-5)
                ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, testval=self.soln['ecc'])
                omega = xo.distributions.Angle("omega")

            # Even-Odd Test
            else:
                logP_even = pm.Normal("logP_even", mu=self.soln['logP_even'], sd=1)
                t0_even = pm.Normal("t0_even", mu=self.soln['t0_even'], sd=1)
                period_even = pm.Deterministic("period_even", tt.exp(logP_even))
                m_star_even = BoundedNormal("m_star_even", mu=self.soln['m_star_even'], sd=M_star_huang[1])
                r_star_even = BoundedNormal("r_star_even", mu=self.soln['r_star_even'], sd=R_star_huang[1])
                b_even = pm.Uniform("b_even", lower=0, upper=0.9, testval=self.soln['b_even'])
                BoundedNormal_logr = pm.Bound(pm.Normal, lower=-5, upper=0)
                logr_even = BoundedNormal_logr('logr_even', mu=self.soln['logr_even'], sd=1.0)
                r_pl_even = pm.Deterministic("r_pl_even", tt.exp(logr_even))
                ror_even = pm.Deterministic("ror_even", r_pl_even / r_star_even)
                BoundedBeta = pm.Bound(pm.Beta, lower=0, upper=1-1e-5)
                ecc_even = BoundedBeta("ecc_even", alpha=0.867, beta=3.03, testval=self.soln['ecc_even'])
                omega_even = xo.distributions.Angle("omega_even")

                logP_odd = pm.Normal("logP_odd", mu=self.soln['logP_odd'], sd=1)
                t0_odd = pm.Normal("t0_odd", mu=self.soln['t0_odd'], sd=1)
                period_odd = pm.Deterministic("period_odd", tt.exp(logP_odd))
                m_star_odd = BoundedNormal("m_star_odd", mu=self.soln['m_star_odd'], sd=M_star_huang[1])
                r_star_odd = BoundedNormal("r_star_odd", mu=self.soln['r_star_odd'], sd=R_star_huang[1])
                b_odd = pm.Uniform("b_odd", lower=0, upper=0.9, testval=self.soln['b_odd'])
                logr_odd = BoundedNormal_logr('logr_odd', mu=self.soln['logr_odd'], sd=1.0)
                r_pl_odd = pm.Deterministic("r_pl_odd", tt.exp(logr_odd))
                ror_odd = pm.Deterministic("ror_odd", r_pl_odd / r_star_odd)
                ecc_odd = BoundedBeta("ecc_odd", alpha=0.867, beta=3.03, testval=self.soln['ecc_odd'])
                omega_odd = xo.distributions.Angle("omega_odd")

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=self.soln['logamp'], sd=5.0)
            logrotperiod = pm.Normal("logrotperiod", mu=self.soln['logrotperiod'], sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=self.soln['logQ0'], sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=self.soln['logdeltaQ'], sd=10.0)
            mix = pm.Uniform("mix", lower=0, upper=1.0, testval=self.soln['mix'])

            # Transit jitter & GP parameters
            logs2 = pm.Normal("logs2", mu=self.soln['logs2'], sd=5.0)

            # Track the rotation period as a deterministic
            rotperiod = pm.Deterministic("rotation_period", tt.exp(logrotperiod))

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
            #burnin = sampler.tune(tune=800, start=self.soln, step_kwargs=dict(target_accept=0.9))
            burnin = sampler.tune(tune=4500, start=self.soln, step_kwargs=dict(target_accept=0.9))
            self.trace = sampler.sample(draws=2000)


    # def assess_conv(self, do_even_odd):
    #     '''
    #     '''
    #     if do_even_odd == False:
    #         pm.summary(self.trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "omega", "ecc", "r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean", "logrotperiod"])
    #     else:
    #         pm.summary(self.trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "mean", "u_star", "logrotperiod", "omega_even", "ecc_even", "r_pl_even", "b_even", "t0_even", "logP_even", "r_star_even", "m_star_even", "omega_odd", "ecc_odd", "r_pl_odd", "b_odd", "t0_odd", "logP_odd", "r_star_odd", "m_star_odd" ])


    # def plot_corner(self, varnames):
    #     '''
    #     '''
    #     #varnames = ["period", "b", "ecc", "r_pl"]
    #     samples = pm.trace_to_dataframe(self.trace, varnames=varnames)
    #
    #     # Convert the radius to Earth radii
    #     samples["r_pl"] = (np.array(samples["r_pl"]) * u.R_sun).to(u.R_earth).value
    #
    #     corner.corner(samples[["period", "r_pl", "b", "ecc"]], labels=["period [days]", "radius [Earth radii]", "impact param", "eccentricity"])
    #     plt.show()
    #     plt.close()






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
