import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import exoplanet as xo
import pickle
import transits


def load_lc(file, directory):
    '''Loads in light curve
    '''
    lc = transits.GetLC(fn=file, fn_dir=directory)

    time = lc.time
    norm_flux = lc.norm_flux
    norm_flux_err = lc.norm_flux_err
    cads = lc.cadences

    return time, norm_flux, norm_flux_err, cads


def plot_norm_lc(tic, time, flux):
    '''Plots normalized light curve
    '''
    plt.plot(time, flux)
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')
    plt.title('Normalized Light Curve TIC{:d}'.format(tic))
    #plt.savefig(fname='/home/earleyn/figures/norm_lc_tic{:d}'.format(tic), dpi=250, format='pdf')
    plt.savefig(fname='/Users/nicholasearley/TESS_data/young_bois_figs/raw_lc_tic{:d}'.format(tic), dpi=250, format='pdf')
    plt.close()


def tot_trns_points(soln):
    ''' Finds total numbers of data points in transit
    '''
    model_lc = np.sum(soln["light_curves"], axis=-1)
    points = 0
    for flux in model_lc:
        if flux != 0:
            points += 1
    return points


def model_lc(results):
    ''' Makes GP model
    '''
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

    return GPmodel, map_soln, mask_out, no_pl_GPmodel, no_pl_map_soln, deltaloglike


def save_data(directory, tic, x_values, y_values, yerr_values, cad_values):
    ''' Saves data loaded in from light curve with pickle
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

    with open(directory+'/data_tic{:d}.p'.format(tic), 'wb') as outfile:
        pickle.dump(data, outfile)



def save_models(directory, tic, deltaloglike_values, planet_results, planet_models, planet_solns, no_pl_models, no_pl_solns, planet_outmasks):
    '''Saves data from modeling with pickle
    '''
    model = {}
    model['Run'] = []
    model['Delta Log Likelihood'] = []
    model['BLS Period'] = []
    model['Planet Results'] = []
    model['Planet Models'] = []
    model['Planet Solns'] = []
    model['No Planet Models'] = []
    model['No Planet Solns'] = []
    model['Masks of Outliers'] = []

    for i, entry in enumerate(deltaloglike_values):
        bls_period = planet_results[i].bls_period
        model['Run'].append(i)
        model['Delta Log Likelihood'].append(entry)
        model['BLS Period'].append(bls_period)
        model['Planet Results'].append(planet_results[i])
        model['Planet Models'].append(planet_models[i])
        model['Planet Solns'].append(planet_solns[i])
        model['No Planet Models'].append(no_pl_models[i])
        model['No Planet Solns'].append(no_pl_solns[i])
        model['Masks of Outliers'].append(planet_outmasks[i])

    with open(directory+'/modeling_results_tic{:d}.p'.format(tic), 'wb') as outfile:
        pickle.dump(model, outfile)



def run_script(file, tic, lc_dir, data_dir):
    '''Runs script
    '''
    # Loading in light curve
    time, norm_flux, norm_flux_err, cads = load_lc(file, lc_dir)

    # Plot and save normalized light curve
    plot_norm_lc(tic, time, norm_flux)

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

    for i in range(10):
        results = transits.FindTransits(time=x_values[i], flux=y_values[i], flux_err=yerr_values[i], cads=cad_values[i], tic=tic, run=i)
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

    for i, result in enumerate(planet_results):
        #GP model plots
        mask = planet_outmasks[i]
        pl_soln = planet_solns[i]
        no_soln = no_pl_solns[i]
        result.plot_lc(soln=pl_soln, mask=mask, pl=True)
        result.plot_lc(soln=no_soln, mask=mask, pl=False)
        result.plot_box()

    save_data(data_dir, tic, x_values, y_values, yerr_values, cad_values)
    save_models(data_dir, tic, deltaloglike_values, planet_results, planet_models, planet_solns, no_pl_models, no_pl_solns, planet_outmasks)


def repeat(tics):
    ''' Finds multi-sector data: same tics, different sector
    '''
    length = len(tics)
    repeated_tics = []

    for i in range(length):
        k = i + 1
        for j in range(k, length):
            if tics[i] == tics[j] and tics[i] not in repeated_tics:
                repeated_tics.append(tics[i])
    return repeated_tics



# Reading files
#lc_dir = '/home/earleyn/lc_young_bois'
#data_dir = '/home/earleyn/data'
lc_dir = '/Users/nicholasearley/TESS_data/young_bois_lc'
data_dir = '/Users/nicholasearley/TESS_data/young_bois_data'
tic_list = []
file_list = []

for filename in os.listdir(lc_dir):
    res_tic = re.findall("_tic(\d+)_", filename)
    tic = int(res_tic[0])
    tic_list.append(tic)
    file_list.append(filename)


multi_tic = repeat(tic_list)
if len(multi_tic) > 0:
    #Multi sector data
    for tic in multi_tic:
        glob_files = glob.glob(lc_dir+'/'+'*tic{}*'.format(tic))
        files = []
        for f in glob_files:
            fixed_fn = f.split(lc_dir+'/')[1]
            files.append(fixed_fn)
        run_script(file=files, tic=tic, lc_dir=lc_dir, data_dir=data_dir)

for i, tic in enumerate(tic_list):
    if (tic not in multi_tic) == True:
        run_script(file=file_list[i], tic=tic, lc_dir=lc_dir, data_dir=data_dir)

# run_script(file=['hlsp_eleanor_tess_ffi_tic232073492_s01_tess_v0.2.2_lc.fits',
#             'hlsp_eleanor_tess_ffi_tic232073492_s02_tess_v0.2.2_lc.fits'], tic=232073492, lc_dir=lc_dir, data_dir=data_dir)
