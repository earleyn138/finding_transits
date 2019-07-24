import matplotlib.pyplot as plt
import numpy as np
import exoplanet as xo
import pymc3 as pm
import os
import re
import transits


# Loading in light curve
directory = '/home/earleyn/lc_young_bois'
for filename in os.listdir(directory):
    res_tic = re.findall("tic(\d+)_", filename)
    tic = int(res_tic[0])
    res_sector = re.findall("s0(\d+)_", filename)
    sector = int(res_sector[0])

    lc = transits.GetLC(fn=filename, fn_dir=directory)

    time = lc.time
    norm_flux = lc.norm_flux
    norm_flux_err = lc.norm_flux_err

    x = time
    y = norm_flux
    yerr = norm_flux_err
    cads = lc.cadences

    plt.plot(x, y)
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')
    plt.title('Normalized Light Curve TIC{:d} Sector{:d}'.format(tic, sector))
    plt.savefig('/home/earleyn/figures/raw_lc_tic{:d}_sect{:d}'.format(tic, sector), dpi=1000)
    plt.clf()

# Total number of points in transit
    def tot_trns_points(soln):
        model_lc = np.sum(soln["light_curves"], axis=-1)
        points = 0
        for flux in model_lc:
            if flux != 0:
                points += 1
        return points

# Results
    x_values = [x]
    y_values = [y]
    yerr_values = [yerr]
    cad_values = [cads]

    deltaloglike_values = []
    planet_results = []
    planet_models = []
    planet_solns = []
    planet_outmasks = []

    no_pl_models = []
    no_pl_solns = []

    for i in range(10):
        results = transits.FindTransits(x_values[i], y_values[i], yerr_values[i], cad_values[i], tic, sector, run=i)

        GPmodel, map_soln0 = results.build_GPmodel()

        # Remove outliers
        mod = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves"], axis=-1)
        resid = results.flux - mod
        rms = np.sqrt(np.median(resid**2))
        mask_out = np.abs(resid) < 5 * rms

        # Rebuild model with outliers masked
        GPmodel, map_soln = results.build_GPmodel(mask_out, map_soln0)

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

        no_pl_models.append(no_pl_GPmodel)
        no_pl_solns.append(no_pl_map_soln)


        # Number of parameters for light curve model: impact parameter, period,
        # limb darkening (2 parameters), t0, radius, eccentricity
        K = 7

        # Total number of data points in transit
        N = tot_trns_points(map_soln)

        if np.abs(deltaloglike) > 0.5 * K * np.log(N):
            # This is a planet
            # Removing this transit to look for another
            y_values.append(y_values[i][mask_out] - np.sum(map_soln["light_curves"], axis=-1))
            x_values.append(x_values[i][mask_out])
            yerr_values.append(yerr_values[i][mask_out])
            cad_values.append(cad_values[i][mask_out])

        else:
            # There are no more planets in the data
            break


    # # Sampling model
    # np.random.seed(42)
    # sampler = xo.PyMC3Sampler(finish=300, chains=4)
    #
    # trace_list = []
    # for GPmodel in planet_models:
    #     with GPmodel:
    #         burnin = sampler.tune(tune=500, start=map_soln, step_kwargs=dict(target_accept=0.9))
    #         trace = sampler.sample(draws=2000)
    #         trace_list.append(trace)
    #
    # pm.summary(trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "omega", "ecc", "r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean", "logrotperiod"])
    # results.plot_corner(trace)
    # results.plot_folded_lc(trace, mask)
