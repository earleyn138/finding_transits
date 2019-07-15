from lightkurve.lightcurve import LightCurve as LC
import pysyzygy as ps


# Injecting transits into light curve
def inject_transit(lk, t0, RpRs, per, exp=0.02):
    trn = ps.Transit(t0 = t0, RpRs = RpRs, per = per)
    flux = trn(lk.time)
    return LC(lk.time, flux*lk.flux)

def multiple_inject(time, norm_flux):
    lk1 = LC(time, norm_flux)

    # First injection
    true_t0 = 1320
    true_RpRs = 0.18
    true_per = 2.4
    inject1 = inject_transit(lk, true_t0, true_RpRs, true_per)

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

    return x, y, yerr
