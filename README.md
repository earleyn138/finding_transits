# finding_transits
This code, to be run on the command line, was designed to find transits around young, rapidly rotating stars. The scripts run.py and short_run.py employ the package transits which contain the two classes GetLC and FindTransits. All together, the basic functionality of the code accomplishes the following:

1) Loads in a normalized light curve with gaps in the data removed
2) Performs initial detrending via Fast Fourier Transform which removes the rotation signal for rapid rotators
3) Carries out the box-least squares method (BLS) on the FFT-detrended light curve to identify initial guesses for the period, time of mid-transit, depth, and duration of a transit
4) Uses the BLS initial guesses to simultaneously build a transit model and Gaussian process model that detrends the initial light curve, prior to FFT-detrending
