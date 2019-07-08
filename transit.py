import matplotlib.pyplot as plt
import numpy as np
import eleanor
import exoplanet as xo
import pymc3 as pm
import os
from bs4 import BeautifulSoup
import requests
from lightkurve.lightcurve import LightCurve as LC
import pysyzygy as ps

import transits


path = requests.get("https://archipelago.uchicago.edu/tess_postcards/young_bois/").text
soup = BeautifulSoup(path, "lxml").find_all('a')

all_url = []
for fn in soup:
    if fn.get('href')[-4::] == 'fits':
        all_url.append(fn.get('href'))

url_list = all_url[32:33]
ffi_dir = '/Users/nicholasearley/TESS_data/detrending'
url_path = 'https://archipelago.uchicago.edu/tess_postcards/young_bois/'

for url in url_list:
    os.system('cd {} && curl -O -L {}'.format(ffi_dir, url_path+url))

file = 'hlsp_eleanor_tess_ffi_tic52284854_s01_tess_v0.1.8rc1_lc.fits'
star = eleanor.Source(fn=file, sector=1, fn_dir=ffi_dir)
data = eleanor.TargetData(star, do_psf=True, do_pca=True)
q = data.quality == 0
raw_flux = data.corr_flux[q]
raw_time = data.time[q]
raw_error = data.flux_err[q]

# Getting rid of gaps in data and normalizing
diff = np.diff(raw_time)
ind  = np.where((diff >= 2*np.std(diff)+np.mean(diff)))[0]

subsets = []
for i in range(len(ind)):
    if i == 0:
        region = np.arange(0, ind[i]+1, 1)
    elif i > 0 and i < (len(ind)-1):
        region = np.arange(ind[i], ind[i+1]+1, 1)
    elif i == (len(ind)-1):
        region = np.arange(ind[i-1], len(raw_time), 1)
    subsets.append(region)
regions = np.array(subsets)

fixed_time = np.array([])
fixed_normflux = np.array([])
fixed_normerror = np.array([])
for reg in regions:
    f = raw_flux[reg]
    t = raw_time[reg]
    err = raw_error[reg]

    fixed_normflux = np.append(f/np.nanmedian(f), fixed_normflux)
    fixed_time = np.append(t, fixed_time)
    fixed_normerror = np.append(err/np.nanmedian(f), fixed_normerror)

fixed_time, fixed_normflux, fixed_normerror = zip(*sorted(zip(fixed_time, fixed_normflux, fixed_normerror)))
fixed_time = np.array(fixed_time)
fixed_normflux = np.array(fixed_normflux)
fixed_normerror = np.array(fixed_normerror)



# Injecting transits into light curve
def inject_transit(lk, t0, RpRs, per, exp=0.02):
    trn = ps.Transit(t0 = t0, RpRs = RpRs, per = per)
    flux = trn(lk.time)
    return LC(lk.time, flux*lk.flux)

lk = LC(fixed_time, fixed_normflux)
true_t0 = 1320
true_RpRs = 0.12
true_per = 3.81
inject1 = inject_transit(lk, true_t0, true_RpRs, true_per)

lk2 = LC(inject1.time, inject1.flux)

# Second injection
true_t0 = 1340
true_RpRs = 0.05
true_per = 2.9
inject2 = inject_transit(lk2, true_t0, true_RpRs, true_per)

lk3 = LC(inject2.time, inject2.flux)

x = lk3.time
y = lk3.flux
yerr = fixed_normerror

# # For when we eventualy deal with real data:
# x = fixed_time
# y = fixed_normflux
# yerr = fixed_normerror

plt.plot(x, y)
plt.legend(loc='upper left', fontsize='xx-small')
plt.ylabel('Normalized Flux')
plt.xlabel('Time')
plt.title('Raw Light Curve')
plt.show()


def find_trns_points(soln):
    model_lc = np.sum(soln["light_curves"], axis=-1)
    points = []

    for i, flux in enumerate(model_lc):
        if flux != 0 and model_lc[i-1] == 0:
            n = 0
            for j in range(30):
                if model_lc[i+j] != 0:
                    n += 1
            points.append(n)

    num_trns_points = np.mean(points)

    return num_trns_points


y_values = [lk3.flux]
planet_models = []
planet_solns = []
planet_results = []
planet_outmasks = []

for i in range(10):
    results = transits.FindTransits(x, y_values[i], yerr)

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

    # Number of parameters for light curve model
    K = 6

    # Avg number of data points in transit
    N = find_trns_points(map_soln)

    if deltaloglike > 0.5 * K * np.log(N):
        # This is a planet
        results.plot_lc(soln=map_soln, mask = mask_out)
        planet_models.append(GPmodel)
        planet_solns.append(map_soln)
        planet_results.append(results)
        planet_outmasks.append(mask_out)

        # Removing this transit to look for another
        y_values.append(y_values[i] - np.sum(map_soln0["light_curves"], axis=-1))

    else:
        # There are no more planets in the data
        break


# # Sampling model
# np.random.seed(42)
# sampler = xo.PyMC3Sampler(finish=300, chains=4)
# with GPmodel:
#     burnin = sampler.tune(tune=500, start=map_soln,
#                           step_kwargs=dict(target_accept=0.9))
#
# with GPmodel:
#     trace = sampler.sample(draws=2000)
#
# pm.summary(trace, varnames=["logamp", "logQ0", "logdeltaQ", "mix", "logs2", "omega", "ecc", "r_pl", "b", "t0", "logP", "r_star", "m_star", "u_star", "mean", "logrotperiod"])
#
# results.plot_folded_lc(trace, mask)
#
# results.plot_corner(trace)
