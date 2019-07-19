import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import eleanor

__all__ = ['GetLC']

class GetLC(object):
    'Uses Stella code to remove gaps in light curve and normalize data'

    def __init__(self, fn=None, fn_dir=None):
        if fn_dir is None:
            self.directory = '.'
        else:
            self.directory = fn_dir

        self.file = fn

        if (type(fn) == list) or (type(fn) == np.ndarray):
            if len(fn) > 1:
                self.multi = True
            else:
                self.multi = False
                self.file  = fn[0]
        else:
            self.multi = False

        self.load_data()
        self.normalize_lc()


    def load_data(self):
        """Allows for the option to pass in multiple files.
        """
        if self.multi is True:
            self.star, self.data = [], []

            for fn in self.file:
                s = eleanor.Source(fn=fn, fn_dir=self.directory, tc=True)
                d = eleanor.TargetData(s)
                self.star.append(s)
                self.data.append(d)
            self.star = np.array(self.star)
            self.data = np.array(self.data)
            self.tic    = self.star[0].tic
            self.coords = self.star[0].coords

        else:
            self.star = eleanor.Source(fn=self.file, fn_dir=self.directory, tc=True)
            self.data = eleanor.TargetData(self.star)
            self.tic  = self.star.tic
            self.coords = self.star.coords

        return


    def find_breaks(self, time=None):
        """Finds gaps due to data downlink or other telescope issues.
        """
        print(len(time))
        if time is None:
            time = self.time
        diff = np.diff(time)
        ind  = np.where((diff >= 2.5*np.std(diff)+np.nanmean(diff)))[0]
        ind = np.append(0, ind)
        ind = np.append(ind, len(time))
        print(ind)

        subsets = []
        for i in range(len(ind)-1):
            region = np.arange(ind[i], ind[i+1], 1)
            print(len(region))
            subsets.append(region)

        return np.array(subsets)



    def normalize_lc(self):
        """Normalizes light curve via chunks of data.
        """
        def normalized_subset(regions, t, flux, err, cads):
            time, norm_flux = np.array([]), np.array([])
            norm_flux_err, cadences = np.array([]), np.array([])

            for reg in regions:
                f          = flux[reg]
                norm_flux  = np.append(f/np.nanmedian(f), norm_flux)
                time       = np.append(t[reg], time)
                e          = err[reg]
                norm_flux_err = np.append(e/np.nanmedian(f), norm_flux_err)
                cadences   = np.append(cads[reg], cadences)
            return time, norm_flux, norm_flux_err, cadences


        self.time, self.norm_flux = np.array([]), np.array([])
        self.norm_flux_err = np.array([])
        self.cadences = np.array([])

        if self.multi is True:
            for d in self.data:
                q = d.quality == 0
                t = d.time[q]
                f = d.corr_flux[q]
                err = d.flux_err[q]

                # Searches for breaks based on differences in time array
                regions = self.find_breaks(time=t)
                sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, t, f, err, d.ffiindex[q])
                self.time = np.append(sector_t, self.time)
                self.norm_flux = np.append(sector_f, self.norm_flux)
                self.norm_flux_err  = np.append(sector_e, self.norm_flux_err)
                self.cadences  = np.append(sector_c, self.cadences)
        else:
            q = self.data.quality == 0
            regions = self.find_breaks(time=self.data.time[q])
            self.regions = regions+0.0
            sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, self.data.time[q],
                                                                       self.data.corr_flux[q],
                                                                       self.data.flux_err[q],
                                                                       self.data.ffiindex[q])
            self.time = sector_t
            print(sector_c)
            self.norm_flux = sector_f
            self.norm_flux_err  = sector_e
            self.cadences  = sector_c

        self.time, self.norm_flux, self.norm_flux_err = zip(*sorted(zip(self.time, self.norm_flux, self.norm_flux_err)))
        self.time, self.norm_flux, self.norm_flux_err = np.array(self.time), np.array(self.norm_flux), np.array(self.norm_flux_err)
        self.cadences = np.sort(self.cadences)
