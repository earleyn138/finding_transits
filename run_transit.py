import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import exoplanet as xo
import pickle
import transits

path = '/home/earleyn/young_bois'
lc_dir = '{}/lc'.format(path)
data_dir = '{}/data'.format(path)
fig_dir = '{}/figures'.format(path)

def load_lc(file, lc_dir):
    '''Loads in light curve
    '''
    lc = transits.GetLC(fn=file, fn_dir=lc_dir)
    time = lc.time
    norm_flux = lc.norm_flux
    norm_flux_err = lc.norm_flux_err
    cads = lc.cadences

    return time, norm_flux, norm_flux_err, cads


def plot_norm_lc(fig_dir, tic, time, flux):
    '''Plots normalized light curve
    '''
    plt.plot(time, flux)
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')
    plt.title('Normalized Light Curve TIC{:d}'.format(tic))
    plt.savefig(fname='{}/norm_lc_tic{:d}'.format(fig_dir, tic), dpi=250, format='pdf')
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
    GPmodel, map_soln = results.build_GPmodel(mask=mask_out, start=map_soln0, pl=True)

    # Model with no planet
    no_pl_GPmodel, no_pl_map_soln = results.build_GPmodel(mask=mask_out, start=map_soln0, pl=False)

    # Log likelihoods
    logp = map_soln['loglikelihood']
    logp0 = no_pl_map_soln['loglikelihood']
    deltaloglike = logp - logp0

    return GPmodel, map_soln, mask_out, no_pl_GPmodel, no_pl_map_soln, deltaloglike


def save_data(data_dir, tic, x_values, y_values, yerr_values, cad_values):
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

    with open(data_dir+'/data_tic{:d}.p'.format(tic), 'wb') as outfile:
        pickle.dump(data, outfile)



def save_models(data_dir, tic, deltaloglike_values, planet_results, planet_models, planet_solns, no_pl_models, no_pl_solns, planet_outmasks):
    '''Saves data from modeling with pickle
    '''
    model = {}
    model['Run'] = []
    model['Delta Log Likelihood'] = []
    model['BLS Period'] = []
    model['BLS T0'] = []
    model['BLS Depth'] = []
    model['BLS Duration'] = []
    model['Planet Results'] = []
    model['Planet Models'] = []
    model['Planet Solns'] = []
    model['No Planet Models'] = []
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
        model['Planet Results'].append(planet_results[i])
        model['Planet Models'].append(planet_models[i])
        model['Planet Solns'].append(planet_solns[i])
        model['No Planet Models'].append(no_pl_models[i])
        model['No Planet Solns'].append(no_pl_solns[i])
        model['Masks of Outliers'].append(planet_outmasks[i])

    with open(data_dir+'/modeling_results_tic{:d}.p'.format(tic), 'wb') as outfile:
        pickle.dump(model, outfile)



def get_results(tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir):
    '''Runs script
    '''
    # Plot and save normalized light curve
    plot_norm_lc(fig_dir, tic, time, norm_flux)

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
                break
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



tic_list = []
file_list = []

for filename in os.listdir(lc_dir):
    res_tic = re.findall("_tic(\d+)_", filename)
    tic = int(res_tic[0])
    #if tic == 373092208:
    tic_list.append(tic)
    file_list.append(filename)

multi_tic = repeat(tic_list)

if len(multi_tic) > 0:
    #Multi sector data
    for tic in multi_tic:
        print('TIC: '+str(tic))
        glob_files = glob.glob(lc_dir+'/'+'*tic{}*'.format(tic))
        files = []
        for f in glob_files:
            fixed_fn = f.split(lc_dir+'/')[1]
            files.append(fixed_fn)
        time, norm_flux, norm_flux_err, cads = load_lc(files, lc_dir)
        if len(time) == 0:
            print('Cannot load in light curve: Skipping tic')
        else:
            get_results(tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir)


for i, tic in enumerate(tic_list):
    if (tic not in multi_tic) == True:
        print('TIC: '+str(tic))
        time, norm_flux, norm_flux_err, cads = load_lc(file_list[i], lc_dir)
        if len(time) == 0:
            print('Cannot load in light curve: Skipping tic')
        else:
            get_results(tic, time, norm_flux, norm_flux_err, cads, data_dir, fig_dir)
