import matplotlib.pyplot as plt
import numpy as np
import exoplanet as xo
import pymc3 as pm
import transits

from lightkurve.lightcurve import LightCurve as LC
import pysyzygy as ps



# Loading in light curve
lc = transits.GetLC(fn=['hlsp_eleanor_tess_ffi_tic139398868_s05_tess_v0.2.1_lc.fits', 'hlsp_eleanor_tess_ffi_tic139398868_s06_tess_v0.2.1_lc.fits'], fn_dir='/Users/nicholasearley/TESS_data/ffi')

time = lc.time
norm_flux = lc.norm_flux
norm_error = lc.flux_err

# Injecting transits into light curve
def inject_transit(lk, t0, RpRs, per, exp=0.02):
    trn = ps.Transit(t0 = t0, RpRs = RpRs, per = per)
    flux = trn(lk.time)
    return LC(lk.time, flux*lk.flux)

lk1 = LC(time, norm_flux)

# First injection
true_t0 = 1320
true_RpRs = 0.18
true_per = 2.4
inject1 = inject_transit(lk1, true_t0, true_RpRs, true_per)

lk2 = LC(inject1.time, inject1.flux)

# Second injection
true_t0 = 1340
true_RpRs = 0.2
true_per = 3.12
inject2 = inject_transit(lk2, true_t0, true_RpRs, true_per)

lk3 = LC(inject2.time, inject2.flux)

# # Third injection
# true_t0 = 1330
# true_RpRs = 0.12
# true_per = 1.9
# inject3 = inject_transit(lk3, true_t0, true_RpRs, true_per)
#
# lk4 = LC(inject3.time, inject3.flux)

x = lk3.time
y = lk3.flux
yerr = norm_error

# # For when we eventualy deal with real data:
# x = time
# y = norm_flux
# yerr = norm_error

plt.plot(x, y)
plt.legend(loc='upper left', fontsize='xx-small')
plt.ylabel('Normalized Flux')
plt.xlabel('Time')
plt.title('Raw Light Curve')
plt.show()

def tot_trns_points(soln):
    model_lc = np.sum(soln["light_curves"], axis=-1)
    points = 0
    for flux in model_lc:
        if flux != 0:
            points += 1
    return points


x_values = [x]
y_values = [y]
yerr_values = [norm_error]

deltaloglike_values = []
planet_results = []
planet_models = []
planet_solns = []
planet_outmasks = []

no_pl_models = []
no_pl_solns = []

for i in range(10):
    results = transits.FindTransits(x_values[i], y_values[i], yerr_values[i])

    GPmodel, map_soln0 = results.build_GPmodel()
    # # Plotting light curves with GP model, transits, residuals before removing outliers
    # results.plot_lc(soln=map_soln0, mask=None)

    # Remove outliers
    mod = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves"], axis=-1)
    resid = results.flux - mod
    rms = np.sqrt(np.median(resid**2))
    mask_out = np.abs(resid) < 5 * rms
    # # Plotting residuals marking outliers
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, resid, "k", label="data")
    # plt.plot(x[~mask], resid[~mask], "xr", label="outliers")
    # plt.axhline(0, color="#aaaaaa", lw=1)
    # plt.ylabel("residuals [ppt]")
    # plt.xlabel("time [days]")
    # plt.legend(fontsize=12, loc=3)
    # plt.xlim(x.min(), x.max())
    # plt.show()

    # Rebuild model with outliers masked
    GPmodel, map_soln = results.build_GPmodel(mask_out, map_soln0)
    # # Plotting light curves with GP model, transits, residuals after removing outliers
    # results.plot_lc(soln=map_soln, mask=mask_out)

    # Model with no planet
    no_pl_GPmodel, no_pl_map_soln = results.build_no_pl_GPmodel(mask_out, map_soln0)

    # Log likelihoods
    logp = map_soln['loglikelihood']
    logp0 = no_pl_map_soln['loglikelihood']
    deltaloglike = logp - logp0
    deltaloglike_values.append(deltaloglike)

    planet_results.append(results)
    planet_models.append(GPmodel)
    planet_solns.append(map_soln)
    planet_outmasks.append(mask_out)
    #####results.plot_lc(soln=map_soln, mask = mask_out)

    no_pl_models.append(no_pl_GPmodel)
    no_pl_solns.append(no_pl_map_soln)


    # Number of parameters for light curve model: impact parameter, period,
    # limb darkening (2 parameters), t0, radius, eccentricity
    K = 7

    # Total number of data points in transit
    N = tot_trns_points(map_soln)

    if deltaloglike > 0.5 * K * np.log(N):
        # This is a planet
        # Removing this transit to look for another
        y_values.append(y_values[i][mask_out] - np.sum(map_soln["light_curves"], axis=-1))
        x_values.append(x_values[i][mask_out])
        yerr_values.append(yerr_values[i][mask_out])

    else:
        # There are no more planets in the data
        break


# Sampling model
np.random.seed(42)
sampler = xo.PyMC3Sampler(finish=300, chains=4)

trace_list = []
for GPmodel in planet_models:
    with GPmodel:
        burnin = sampler.tune(tune=500, start=map_soln, step_kwargs=dict(target_accept=0.9))
        trace = sampler.sample(draws=2000)
        trace_list.append(trace)

# pm.summary(trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "omega", "ecc", "r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean", "logrotperiod"])
#results.plot_corner(trace)
#results.plot_folded_lc(trace, mask)
