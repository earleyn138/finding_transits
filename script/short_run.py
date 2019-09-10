import os
import re
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import exoplanet as xo
import pickle
import transits

# Relevant TIC
rel_tic = int(sys.argv[1])
print('TIC: '+str(rel_tic))

# Path to the project directory containing subdirectories for light curves, data, and figures
path = sys.argv[2]

# run_type
if sys.argv[3] == 'do_even_odd':
    do_even_odd = True
    run_all = False
    do_EB = False
elif sys.argv[3] == 'run_all':
    do_even_odd = False
    run_all = True
    do_EB = False
elif sys.argv[3] == 'EB':
    do_even_odd = False
    run_all = False
    do_EB = True


# Light curves to be analyzed: Either Corrected flux, PSF flux, or Raw flux
if sys.argv[4] == 'do_corr':
    do_corr = True
    do_psf = False
    do_raw = False
elif sys.argv[4] == 'do_psf':
    do_corr = False
    do_psf = True
    do_raw = False
elif sys.argv[4] == 'do_raw':
    do_corr = False
    do_psf = False
    do_raw = True

# Using pre-existing files to load in data, or getting light curves with tic ID
if sys.argv[5] == 'use_files':
    use_files = True
elif sys.argv[5] == 'use_tics':
    use_files = False

# Light curve subdirectory
lc_dir = '{}lc'.format(path)

# Data and figures subdirectories
if do_corr == True:
    data_dir = '{}data'.format(path)
    fig_dir = '{}figures'.format(path)
elif do_psf == True:
    data_dir = '{}psf_data'.format(path)
    fig_dir = '{}psf_figures'.format(path)
elif do_raw == True:
    data_dir = '{}raw_data'.format(path)
    fig_dir = '{}raw_figures'.format(path)

# Subdirectory for even-odd test results
vet_dir = '{}even_odd'.format(path)



# Functions for loading in light curves, building models, and saving results

def load_lc(file=None, tic=None, lc_dir=None, do_corr=do_corr, do_raw=do_raw, do_psf=do_psf, use_files=use_files):
    '''Loads in light curve
    '''

    if do_corr == True:
        lc = transits.GetLC(fn=file, fn_dir=lc_dir, tic=tic, use_files=use_files)
    elif do_raw == True:
        lc = transits.GetLC(fn=file, fn_dir=lc_dir, tic=tic, use_files=use_files, do_corr=False, do_raw=True, do_psf=False)
    elif do_psf == True:
        lc = transits.GetLC(fn=file, fn_dir=lc_dir, tic=tic, use_files=use_files, do_corr=False, do_raw=False, do_psf=True)

    time = lc.time
    norm_flux = lc.norm_flux
    norm_flux_err = lc.norm_flux_err
    cads = lc.cadences

    del(lc)

    return time, norm_flux, norm_flux_err, cads



def plot_norm_lc(fig_dir, tic, time, flux, vet=False):
    '''Plots normalized light curve
    '''
    plt.plot(time, flux)
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')
    plt.title('Normalized Light Curve TIC{:d}'.format(tic))
    if vet == False:
        plt.savefig(fname='{}/norm_lc_tic{:d}'.format(fig_dir, tic), dpi=250, format='pdf')
    plt.close()


def tot_trns_points(soln, vet=False, EB=False):
    ''' Finds total numbers of data points in transit
    '''
    if vet == False and EB == False:
        model_lc = np.sum(soln["light_curves"], axis=-1)
        points = 0
        for flux in model_lc:
            if flux != 0:
                points += 1

    elif vet == True and EB == False:
        model_lc = np.sum(soln["light_curves_even"], axis=-1) + np.sum(soln["light_curves_odd"], axis=-1)
        points = 0
        for flux in model_lc:
            if flux != 0:
                points += 1

    elif vet == False and EB == True:
        model_lc = np.sum(soln["light_curves_1"], axis=-1) + np.sum(soln["light_curves_2"], axis=-1)
        points = 0
        for flux in model_lc:
            if flux != 0:
                points += 1

    return points


def model_lc(results, vet=False, EB=False):
    ''' Makes GP model
    '''
    GPmodel, map_soln0 = results.build_GPmodel()

    # Remove outliers
    if vet == False and EB == False:
        mod = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves"], axis=-1)
        resid = results.flux - mod
        rms = np.sqrt(np.median(resid**2))
        mask_out = np.abs(resid) < 5 * rms

    elif vet == True and EB == False:
        mod_even = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves_even"], axis=-1)
        mod_odd = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves_odd"], axis=-1)
        mod = mod_even + mod_odd
        resid = results.flux - mod
        rms = np.sqrt(np.median(resid**2))
        mask_out = np.abs(resid) < 5 * rms

    elif vet == False and EB == True:
        mod_1 = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves_1"], axis=-1)
        mod_2 = map_soln0["pred"] + map_soln0["mean"] + np.sum(map_soln0["light_curves_2"], axis=-1)
        mod = mod_1 + mod_2
        resid = results.flux - mod
        rms = np.sqrt(np.median(resid**2))
        mask_out = np.abs(resid) < 5 * rms


    # Rebuild model with outliers masked
    GPmodel, map_soln = results.build_GPmodel(mask=mask_out, start=map_soln0, pl=True)

    # Model with no planet
    no_pl_GPmodel, no_pl_map_soln = results.build_GPmodel(mask=mask_out, start=map_soln0, pl=False)

    # Log likelihoods
    logp = map_soln['loglikelihood']
    logp0 = no_pl_map_soln['loglikelihood']
    deltaloglike = logp - logp0

    return GPmodel, map_soln, mask_out, no_pl_GPmodel, no_pl_map_soln, deltaloglike


def save_data(data_dir, tic, x_values, y_values, yerr_values, cad_values, vet=False, vet_dir='.'):
    ''' Saves data loaded in from light curve, using pickle
    '''
    data = {}
    data['Run'] = []
    data['Time'] = []
    data['Normalized Flux'] = []
    data['Normalized Error'] = []
    data['Cadences'] = []

    for i, entry in enumerate(x_values):
        data['Run'].append(i)
        data['Time'].append(entry)
        data['Normalized Flux'].append(y_values[i])
        data['Normalized Error'].append(yerr_values[i])
        data['Cadences'].append(cad_values[i])

    if vet == False:
        with open(data_dir+'/data_tic{:d}.p'.format(tic), 'wb') as outfile:
            pickle.dump(data, outfile)

    else:
        with open(vet_dir+'/vet_data_tic{:d}.p'.format(tic), 'wb') as outfile:
            pickle.dump(data, outfile)



def save_models(data_dir, tic, planet_results, deltaloglike_values, planet_solns, no_pl_solns, planet_outmasks, vet=False, vet_dir='.'):
    '''Saves data from modeling, using pickle
    '''
    model = {}
    model['Run'] = []
    model['Delta Log Likelihood'] = []
    model['BLS Period'] = []
    model['BLS T0'] = []
    model['BLS Depth'] = []
    model['BLS Duration'] = []
    model['Planet Solns'] = []
    model['No Planet Solns'] = []
    model['Masks of Outliers'] = []

    for i, entry in enumerate(deltaloglike_values):
        bls_period = planet_results[i].bls_period
        bls_t0 = planet_results[i].bls_t0
        bls_depth = planet_results[i].bls_depth
        bls_duration = planet_results[i].bls_duration
        model['Run'].append(i)
        model['Delta Log Likelihood'].append(entry)
        model['BLS Period'].append(bls_period)
        model['BLS T0'].append(bls_t0)
        model['BLS Depth'].append(bls_depth)
        model['BLS Duration'].append(bls_duration)
        model['Planet Solns'].append(planet_solns[i])
        model['No Planet Solns'].append(no_pl_solns[i])
        model['Masks of Outliers'].append(planet_outmasks[i])

    if vet == False:
        with open(data_dir+'/modeling_results_tic{:d}.p'.format(tic), 'wb') as outfile:
            pickle.dump(model, outfile)

    else:
        with open(vet_dir+'/vet_modeling_results_tic{:d}.p'.format(tic), 'wb') as outfile:
            pickle.dump(model, outfile)



def get_results(tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir, vet=False, vet_dir='.', EB=False):
    '''Runs script, builds models
    '''
    # Plot and save normalized light curve
    if vet == False:
        plot_norm_lc(fig_dir, tic, time, norm_flux)

    else:
        plot_norm_lc(fig_dir, tic, time, norm_flux, vet)

    # Results of modeling
    x_values = [time]
    y_values = [norm_flux]
    yerr_values = [norm_flux_err]
    cad_values = [cads]

    planet_results = []
    planet_models = []
    planet_solns = []
    planet_outmasks = []
    deltaloglike_values = []

    no_pl_models = []
    no_pl_solns = []

    if vet == False and EB == False:
        for i in range(10):
            results = transits.FindTransits(fig_dir=fig_dir, time=x_values[i], flux=y_values[i], flux_err=yerr_values[i], cads=cad_values[i], tic=tic, run=i)

            planet_results.append(results)

            GPmodel, map_soln, mask_out, no_pl_GPmodel, no_pl_map_soln, deltaloglike = model_lc(results)
            planet_models.append(GPmodel)
            planet_solns.append(map_soln)
            planet_outmasks.append(mask_out)
            no_pl_models.append(no_pl_GPmodel)
            no_pl_solns.append(no_pl_map_soln)
            deltaloglike_values.append(deltaloglike)

            # Number of parameters for light curve model: impact parameter, period,
            # limb darkening (2 parameters), t0, radius, eccentricity
            K = 7

            # Total number of data points in transit
            N = tot_trns_points(map_soln)
            if N > 0:
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
            else:
                break

    # Even-odd
    elif vet == True and EB == False:
        for i in range(10):
            results = transits.FindTransits(fig_dir=fig_dir, time=x_values[i], flux=y_values[i], flux_err=yerr_values[i], cads=cad_values[i], tic=tic, run=i, vet=vet, vet_dir=vet_dir)

            planet_results.append(results)

            GPmodel, map_soln, mask_out, no_pl_GPmodel, no_pl_map_soln, deltaloglike = model_lc(results=results, vet=vet)
            planet_models.append(GPmodel)
            planet_solns.append(map_soln)
            planet_outmasks.append(mask_out)
            no_pl_models.append(no_pl_GPmodel)
            no_pl_solns.append(no_pl_map_soln)
            deltaloglike_values.append(deltaloglike)

            # Number of parameters for light curve model: impact parameter, period,
            # limb darkening (2 parameters), t0, radius, eccentricity
            K = 7

            # Total number of data points in transit
            N = tot_trns_points(soln=map_soln, vet=vet)
            if N > 0:
                if np.abs(deltaloglike) > 0.5 * K * np.log(N):
                    # This is a planet
                    # Removing this transit to look for another
                    y_values.append(y_values[i][mask_out] - (np.sum(map_soln["light_curves_even"], axis=-1) + np.sum(map_soln["light_curves_odd"], axis=-1)))
                    x_values.append(x_values[i][mask_out])
                    yerr_values.append(yerr_values[i][mask_out])
                    cad_values.append(cad_values[i][mask_out])

                else:
                    # There are no more planets in the data
                    break
            else:
                break


    elif vet == False and EB == True:
        for i in range(10):
            results = transits.FindTransits(fig_dir=fig_dir, time=x_values[i], flux=y_values[i], flux_err=yerr_values[i], cads=cad_values[i], tic=tic, run=i, EB=EB)

            planet_results.append(results)

            GPmodel, map_soln, mask_out, no_pl_GPmodel, no_pl_map_soln, deltaloglike = model_lc(results=results, EB=EB)
            planet_models.append(GPmodel)
            planet_solns.append(map_soln)
            planet_outmasks.append(mask_out)
            no_pl_models.append(no_pl_GPmodel)
            no_pl_solns.append(no_pl_map_soln)
            deltaloglike_values.append(deltaloglike)

            # Number of parameters for light curve model: impact parameter, period,
            # limb darkening (2 parameters), t0, radius, eccentricity
            K = 7

            # Total number of data points in transit
            N = tot_trns_points(soln=map_soln, EB=EB)
            if N > 0:
                if np.abs(deltaloglike) > 0.5 * K * np.log(N):
                    y_values.append(y_values[i][mask_out] - (np.sum(map_soln["light_curves_1"], axis=-1) + np.sum(map_soln["light_curves_2"], axis=-1)))
                    x_values.append(x_values[i][mask_out])
                    yerr_values.append(yerr_values[i][mask_out])
                    cad_values.append(cad_values[i][mask_out])

                else:
                    break

            else:
                break


    for i, result in enumerate(planet_results):
        #GP model plots
        mask = planet_outmasks[i]
        pl_soln = planet_solns[i]
        no_soln = no_pl_solns[i]
        result.plot_lc(soln=pl_soln, mask=mask, pl=True)
        result.plot_lc(soln=no_soln, mask=mask, pl=False)
        result.plot_box()

    save_data(data_dir, tic, x_values, y_values, yerr_values, cad_values, vet, vet_dir)
    save_models(data_dir, tic, planet_results, deltaloglike_values, planet_solns, no_pl_solns, planet_outmasks, vet, vet_dir)


# Running script
# Using file names to load in lc
if use_files == True:
    glob_files = glob.glob(lc_dir+'/'+'*tic{}*'.format(rel_tic))
    files = []
    for f in glob_files:
        fixed_fn = f.split(lc_dir+'/')[1]
        files.append(fixed_fn)
    time, norm_flux, norm_flux_err, cads = load_lc(file=files, lc_dir=lc_dir, do_corr=do_corr, do_raw=do_raw, do_psf=do_psf, use_files=use_files)

## Using tic to load in lc
else:
    time, norm_flux, norm_flux_err, cads = load_lc(tic=rel_tic, do_corr=do_corr, do_raw=do_raw, do_psf=do_psf)

if len(time) == 0:
    print('Cannot load in light curve: Skipping tic')
else:
    if run_all == True:
        get_results(rel_tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir)
    elif do_even_odd == True:
        get_results(rel_tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir, vet=True, vet_dir=vet_dir)
    elif do_EB == True:
        get_results(rel_tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir, EB=True)
