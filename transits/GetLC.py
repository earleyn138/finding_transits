import numpy as np
import astropy.units as u

import eleanor

__all__ = ['GetLC']

class GetLC(object):
    'Uses Stella code to remove gaps in light curve and normalize data'
    # Using filenames
    def __init__(self, fn=None, fn_dir=None, tic=None, use_files=True, do_corr=True, do_raw= False, do_psf=False):

        self.use_files = use_files

        if self.use_files == True:
            if fn_dir is None:
                self.directory = '.'
            else:
                self.directory = fn_dir

            self.file = fn
            self.do_corr = do_corr
            self.do_raw = do_raw
            self.do_psf = do_psf

            if (type(fn) == list) or (type(fn) == np.ndarray):
                if len(fn) > 1:
                    self.multi = True
                else:
                    self.multi = False
                    self.file  = fn[0]
            else:
                self.multi = False

        else:
            self.tic = tic
            self.do_corr = do_corr
            self.do_raw = do_raw
            self.do_psf = do_psf

            self.star_results = eleanor.multi_sectors(tic=self.tic, sectors='all', tc=True)
            if len(self.star_results) > 1:
                self.multi = True
            else:
                self.multi = False

        self.load_data()
        self.normalize_lc()


    def load_data(self):
        """Allows for the option to pass in multiple files.
        """
        if self.use_files == True:
            if self.multi is True:
                self.star, self.data = [], []
                for fn in self.file:
                    s = eleanor.Source(fn=fn, fn_dir=self.directory, tc=True)
                    if self.do_corr == True or self.do_raw == True:
                        d = eleanor.TargetData(s)
                    else:
                        d = eleanor.TargetData(s, do_psf=True)
                    self.star.append(s)
                    self.data.append(d)
                self.star = np.array(self.star)
                self.data = np.array(self.data)
                self.tic    = self.star[0].tic
                self.coords = self.star[0].coords

            else:
                self.star = eleanor.Source(fn=self.file, fn_dir=self.directory, tc=True)
                if self.do_corr == True or self.do_raw == True:
                    self.data = eleanor.TargetData(self.star)
                else:
                    self.data = eleanor.TargetData(self.star, do_psf=True)
                self.tic  = self.star.tic
                self.coords = self.star.coords

        else:
            if self.multi is True:
                self.star, self.data = [], []
                for s in self.star_results:
                    self.star.append(s)
                    if self.do_corr == True or self.do_raw == True:
                        d = eleanor.TargetData(s)
                    else:
                        d = eleanor.TargetData(s, do_psf=True)
                    self.data.append(d)

            else:
                self.star = self.star_results[0]
                if self.do_corr == True or self.do_raw == True:
                    self.data = eleanor.TargetData(self.star)
                else:
                    self.data = eleanor.TargetData(self.star, do_psf=True)

        return

    def find_breaks(self, time=None):
        """Finds gaps due to data downlink or other telescope issues.
        """
        if time is None:
            time = self.time
        diff = np.diff(time)
        ind  = np.where((diff >= 2.5*np.std(diff)+np.nanmean(diff)))[0]
        ind = np.append(0, ind)
        ind = np.append(ind, len(time))

        subsets = []
        for i in range(len(ind)-1):
            region = np.arange(ind[i], ind[i+1], 1)
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
                if np.nanmedian(f) > 0:
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
                if self.do_corr == True:
                    f = d.corr_flux[q]

                elif self.do_raw == True:
                    f = d.raw_flux[q]

                elif self.do_psf == True:
                    f = d.psf_flux[q]

                err = d.flux_err[q]

                # Searches for breaks based on differences in time array
                regions = self.find_breaks(time=t)
                sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, t, f, err, np.array(d.ffiindex)[q])
                self.time = np.append(sector_t, self.time)
                self.norm_flux = np.append(sector_f, self.norm_flux)
                self.norm_flux_err  = np.append(sector_e, self.norm_flux_err)
                self.cadences  = np.append(sector_c, self.cadences)
        else:
            q = self.data.quality == 0
            regions = self.find_breaks(time=self.data.time[q])
            self.regions = regions+0.0

            if self.do_corr == True:
                sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, self.data.time[q], self.data.corr_flux[q], self.data.flux_err[q], np.array(self.data.ffiindex)[q])

            elif self.do_raw == True:
                sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, self.data.time[q], self.data.raw_flux[q], self.data.flux_err[q], np.array(self.data.ffiindex)[q])

            elif self.do_psf == True:
                sector_t, sector_f, sector_e, sector_c = normalized_subset(regions, self.data.time[q], self.data.psf_flux[q], self.data.flux_err[q], np.array(self.data.ffiindex)[q])

            self.time = sector_t
            self.norm_flux = sector_f
            self.norm_flux_err  = sector_e
            self.cadences  = sector_c


        if len(self.time) > 0:
            self.time, self.norm_flux, self.norm_flux_err = zip(*sorted(zip(self.time, self.norm_flux, self.norm_flux_err)))
            self.time, self.norm_flux, self.norm_flux_err = np.array(self.time), np.array(self.norm_flux), np.array(self.norm_flux_err)
            self.cadences = np.sort(self.cadences)
