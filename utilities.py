import numpy as np
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import tarfile as _tarfile
from beatlas.be_theory import W2oblat, oblat2w, beta
from beatlas.constants import *
from beatlas.PT_makemag import convolution_Johnsons, convolution_JohnsonsZP
from PyAstronomy import pyasl
from scipy.signal import detrend
from astropy.stats import median_absolute_deviation as MAD
from scipy.optimize import curve_fit
from icecream import ic

def norm_n0s(M, num=8, n0min=1e11):
    """ doc 
    """
    n0max = np.exp(0.14*M+31.02)
    if M <= 7.:
        n0max = np.exp(-0.152*M**2+2.51*M+22.09)
    return np.logspace(np.log10(n0min), np.log10(n0max), num=num)

def kde_scipy(x, x_grid, bandwidth=0.2):
    """Kernel Density Estimation with Scipy"""

    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    return kde.evaluate(x_grid)


def set_ranges(INCLUDE_RV, PLX, D_PLX, POL, listpar):
    if INCLUDE_RV:
        ebmv, rv = [[0.0, 0.1], [2.0, 5.8]]
    else:
        rv = 3.1
        ebmv, rv = [[0.0, 0.03], None]

    # dist_min = 0.01
    # dist_max = 10.0
    dist_min = PLX - (3.0 * D_PLX)
    dist_max = PLX + (3.0 * D_PLX)
    if dist_min < 0.0:
        dist_min = 0.0

    ranges = []
    for val in listpar:
        ranges.append([np.min(val), np.max(val)])

    if not POL:
        ranges.append([dist_min, dist_max])
        ranges.append([ebmv[0], ebmv[-1]])

    if INCLUDE_RV:
        ranges.append(rv)

    Ndim = len(ranges)

    return np.array(ranges), Ndim


def find_lim(SED, INCLUDE_RV, MODEL):
    """ Defines the value of "lim", to only use the model params in the
    interpolation

    Usage:
    lim = find_lim()

    """
    if SED:
        if INCLUDE_RV:
            lim = 3
        else:
            lim = 2
    else:
        if MODEL == "BeAtlas2022_phot":
            lim = -4
        else:
            lim = -7
    return lim


def bin_data(x, y, nbins, xran=None, exclude_empty=True):
    """
    Bins data

    Usage:
    xbin, ybin, dybin = bin_data(x, y, nbins, xran=None, exclude_empty=True)

    where dybin is the standard deviation inside the bins.
    """
    # make sure it is a numpy array
    x = np.array([x]).reshape((-1))
    y = np.array([y]).reshape((-1))
    # make sure it is in increasing order
    ordem = x.argsort()
    x = x[ordem]
    y = y[ordem]

    if xran is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = xran[0], xran[1]

    xborders = np.linspace(xmin, xmax, nbins + 1)
    xbin = 0.5 * (xborders[:-1] + xborders[1:])

    ybin = np.zeros(nbins)
    dybin = np.zeros(nbins)
    for i in range(nbins):
        aux = (x > xborders[i]) * (x < xborders[i + 1])
        if np.array([aux]).any():
            ybin[i] = np.mean(y[aux])
            dybin[i] = np.std(y[aux])
        else:
            ybin[i] = np.nan
            dybin[i] = np.nan

    if exclude_empty:
        keep = np.logical_not(np.isnan(ybin))
        xbin, ybin, dybin = xbin[keep], ybin[keep], dybin[keep]

    return xbin, ybin, dybin


def jy2cgs(flux, lbd, inverse=False):
    """
    Converts from Jy units to erg/s/cm2/micron, and vice-versa

    [lbd] = micron

    Usage:
    flux_cgs = jy2cgs(flux, lbd, inverse=False)
    """
    if not inverse:
        flux_new = 3e-9 * flux / lbd ** 2
    else:
        flux_new = lbd ** 2 * flux / 3e-9

    return flux_new


# def find_nearest(array, value):
#     """
#     Find the nearest value inside an array
#     """
#
#     idx = (np.abs(array - value)).argmin()
#     return array[idx], idx


def find_nearest(array, value, bigger=None, idx=False):
    """ Find nearest VALUE in the array and return it.

    INPUT: array, value

    OUTPUT: closest value (array dtype)
    """
    if bigger is None:
        array = np.array(array)
        i = (np.abs(array - value)).argmin()
        found = array[i]
    elif bigger:
        found = np.min([x for x in array if x > value])
        i = np.where(array == found)
    elif not bigger:
        found = np.max([x for x in array if x < value])
        i = np.where(array == found)
    # else:
    # _warn.warn("# ERROR at bigger!!")
    # return
    if not idx:
        return found
    else:
        return i


# ==============================================================================
def find_neighbours(par, par_grid, ranges):
    """
    Finds neighbours' positions of par in par_grid.

    Usage:
    keep, out, inside_ranges, par_new, par_grid_new = \
        find_neighbours(par, par_grid, ranges):

    where redundant columns in 'new' values are excluded,
    but length is preserved (i.e., par_grid[keep] in griddata call).
    """
    # check if inside ranges

    if len(par) == 4:
        ranges = ranges[0:4]
    if len(par) == 3:
        ranges = ranges[0:3]
    # print(par, len(ranges))
    # print(par, ranges)
    count = 0
    inside_ranges = True
    while (inside_ranges is True) * (count < len(par)):
        inside_ranges = (par[count] >= ranges[count, 0]) * (
            par[count] <= ranges[count, 1]
        )
        count += 1

    # find neighbours
    keep = np.array(len(par_grid) * [True])
    out = []

    if inside_ranges:
        for i in range(len(par)):
            # coincidence
            if (par[i] == par_grid[:, i]).any():
                keep *= par[i] == par_grid[:, i]
                out.append(i)
            # is inside
            else:
                # list of values
                par_list = np.array(list(set(par_grid[:, i])))
                # nearest value at left
                par_left = par_list[par_list < par[i]]
                par_left = par_left[np.abs(par_left - par[i]).argmin()]
                # nearest value at right
                par_right = par_list[par_list > par[i]]
                par_right = par_right[np.abs(par_right - par[i]).argmin()]
                # select rows
                kl = par_grid[:, i] == par_left
                kr = par_grid[:, i] == par_right
                keep *= kl + kr
        # delete coincidences
        par_new = np.delete(par, out)
        par_grid_new = np.delete(par_grid, out, axis=1)
    else:
        print("Warning: parameter outside ranges.")
        par_new = par
        par_grid_new = par_grid

    return keep, out, inside_ranges, par_new, par_grid_new


def geneva_interp_fast(Mstar, oblat, t, Zstr="014", silent=True):
    """
    Interpolates Geneva stellar models, from grid of
    pre-computed interpolations.

    Usage:
    Rpole, logL, age = geneva_interp_fast(Mstar, oblat, t, Zstr='014')

    where t is given in tMS, and tar is the open tar file. For now, only
    Zstr='014' is available.
    """
    # read grid
    # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
    dir0 = "../defs/geneve_models/"
    if Mstar <= 20.0:
        fname = "geneva_interp_Z{:}.npz".format(Zstr)
    else:
        fname = "geneva_interp_Z{:}_highM.npz".format(Zstr)
    data = np.load(dir0 + fname)
    Mstar_arr = data["Mstar_arr"]
    oblat_arr = data["oblat_arr"]
    t_arr = data["t_arr"]
    Rpole_grid = data["Rpole_grid"]
    logL_grid = data["logL_grid"]
    age_grid = data["age_grid"]

    # build grid of parameters
    par_grid = []
    for M in Mstar_arr:
        for ob in oblat_arr:
            for tt in t_arr:
                par_grid.append([M, ob, tt])
    par_grid = np.array(par_grid)

    # set input/output parameters
    par = np.array([Mstar, oblat, t])

    # set ranges
    ranges = np.array(
        [[par_grid[:, i].min(), par_grid[:, i].max()] for i in range(len(par))]
    )

    # find neighbours
    keep, out, inside_ranges, par, par_grid = find_neighbours(par, par_grid, ranges)

    # interpolation method
    if inside_ranges:
        interp_method = "linear"
    else:
        if not silent:
            print(
                "[geneva_interp_fast] Warning: parameters out of available range, taking closest model"
            )
        interp_method = "nearest"

    if len(keep[keep]) == 1:
        # coincidence
        Rpole = Rpole_grid.flatten()[keep][0]
        logL = logL_grid.flatten()[keep][0]
        age = age_grid.flatten()[keep][0]
    else:
        # interpolation
        Rpole = griddata(
            par_grid[keep],
            Rpole_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]
        logL = griddata(
            par_grid[keep],
            logL_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]
        age = griddata(
            par_grid[keep],
            age_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]

    return Rpole, logL, age


def geneva_interp(Mstar, oblat, t, Zstr="014", tar=None, silent=True):
    """
    Interpolates Geneva stellar models.

    Usage:
    Rpole, logL, age = geneva_interp(Mstar, oblat, t, tar=None, silent=True)

    where t is given in tMS, and tar is the open tar file. The chosen
    metallicity is according to the input tar file. If tar=None, the
    code will take Zstr='014' by default.
    """
    # oblat to Omega/Omega_c
    # w = oblat2w(oblat)

    # grid
    if Mstar <= 20.0:
        Mlist = np.array([1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0])
    else:
        Mlist = np.array([20.0, 25.0, 32.0, 40.0, 60.0, 85.0, 120.0])

    # read tar file
    if tar is None:
        # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
        dir0 = "../defs/geneve_models/"
        fmod = "Z{:}.tar.gz".format(Zstr)
        tar = _tarfile.open(dir0 + fmod, "r:gz")
    else:
        Zstr = tar.getnames()[0][7:10]

    # interpolation

    # creation of lists for polar radius and log, for extrapolation fit
    # and the ttms list used in the linear fit originally

    ttms = [0, 0.40, 0.65, 0.85, 1.00]
    L_log = []
    Rp = []

    # for ages inside the original grid, nothing happens
    if (t <= 1.0) * (t >= 0.0):
        Rpole, logL, age = geneva_interp_fast(Mstar, oblat, t, Zstr=Zstr, silent=silent)

        return Rpole, logL, age

    if t > 1.0:
        for time in ttms:
            Rpole, logL, age = geneva_interp_fast(
                Mstar, oblat, time, Zstr=Zstr, silent=silent
            )

            Rp.append(Rpole)
            L_log.append(logL)

        coeffs = np.polyfit(np.log10(ttms[-4:]), np.log10(Rp[-4:]), deg=1)
        poly = np.poly1d(coeffs)

        coeffs2 = np.polyfit(ttms[-4:], L_log[-4:], deg=1)
        poly2 = np.poly1d(coeffs2)

        Rpole = 10 ** (poly(np.log10(t)))
        logL = poly2(t)

        # in this case, no age. no physical meaning!
        return Rpole, logL


def griddataBA(minfo, models, params, listpar, dims):
    """
    Moser's routine to interpolate BeAtlas models
    obs: last argument ('listpar') had to be included here
    """

    # ic(params[0])
    # ic(listpar)
    idx = np.arange(len(minfo))
    lim_vals = len(params) * [
        [],
    ]
    for i in range(len(params)):
        lim_vals[i] = [
            find_nearest(listpar[i], params[i], bigger=False),
            find_nearest(listpar[i], params[i], bigger=True),
        ]
        # ic(lim_vals[i])

        tmp = np.where(
            (minfo[:, i] == lim_vals[i][0]) | (minfo[:, i] == lim_vals[i][1])
        )

        idx = np.intersect1d(idx, tmp[0])
        # ic(idx)

    # print(idx)
    # ic(len(minfo[0]))
    out_interp = griddata(minfo[idx], models[idx], params)[0]

    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:

        mdist = np.zeros(np.shape(minfo))
        ichk = range(len(params))
        for i in ichk:
            mdist[:, i] = np.abs(minfo[:, i] - params[i]) / (
                np.max(listpar[i]) - np.min(listpar[i])
            )
        idx = np.where(np.sum(mdist, axis=1) == np.min(np.sum(mdist, axis=1)))
        if len(idx[0]) != 1:
            out_interp = griddata(minfo[idx], models[idx], params)[0]
        else:
            out_interp = models[idx][0]

    return out_interp


# ==============================================================================
def griddataBAtlas(minfo, models, params, listpar, dims, isig):
    idx = range(len(minfo))
    lim_vals = len(params) * [
        [],
    ]
    for i in [i for i in range(len(params)) if i != isig]:
        lim_vals[i] = [
            find_nearest(listpar[i], params[i], bigger=False),
            find_nearest(listpar[i], params[i], bigger=True),
        ]
        tmp = np.where(
            (minfo[:, i] == lim_vals[i][0]) | (minfo[:, i] == lim_vals[i][1])
        )
        idx = np.intersect1d(idx, tmp)
        #
    out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:
        idx = np.arange(len(minfo))
        for i in [i for i in range(len(params)) if i != dims["sig0"]]:
            imin = lim_vals[i][0]
            if lim_vals[i][0] != np.min(listpar[i]):
                imin = find_nearest(listpar[i], lim_vals[i][0], bigger=False)
            imax = lim_vals[i][1]
            if lim_vals[i][1] != np.max(listpar[i]):
                imax = find_nearest(listpar[i], lim_vals[i][1], bigger=True)
            lim_vals[i] = [imin, imax]
            tmp = np.where(
                (minfo[:, i] >= lim_vals[i][0]) & (minfo[:, i] <= lim_vals[i][1])
            )
            idx = np.intersect1d(idx, tmp.flatten())
        out_interp = griddata(minfo[idx], models[idx], params)[0]

    return out_interp


def print_to_latex(MODEL, LABELS2, fname, params_fit, errors_fit):
    """
    Prints results in latex table format

    """

    file1 = open(fname + "_TABLE.txt", "w")
    L = [
        r"\begin{table}" + " \n",
        "\centering \n",
        r"\begin{tabular}{lll}" + " \n",
        "\hline \n",
        "Parameter  & Value & Type \\\ \n",
        "\hline \n",
    ]
    file1.writelines(L)

    params_to_print = []
    # print(errors_fit[0][1])
    for i in range(len(params_fit)):
        params_to_print.append(
            LABELS2[i]
            + "= {0:.3f} +{1:.3f} -{2:.3f}".format(
                params_fit[i], errors_fit[i][0], errors_fit[i][1]
            )
        )
        file1.writelines(
            LABELS2[i]
            + "& ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Free \\\ \n".format(
                params_fit[i], errors_fit[i][0], errors_fit[i][1]
            )
        )

    # if len(hpds[0]) > 1:

    if MODEL == "BeAtlas2022_phot":
        Mstar = params_fit[0]
        W = params_fit[1]
        tms = params_fit[2]
        incl = params_fit[3]
        incl_range = [incl + errors_fit[3][0], incl - errors_fit[3][1]]
        # dist = best_pars[4][0]
        # ebmv = best_pars[5][0]
    if MODEL == "acol" or MODEL == "aara" or MODEL == "pol" or MODEL == "BeAtlas2022_disk":
        Mstar = params_fit[0]
        W = params_fit[1]
        tms = params_fit[2]
        incl = params_fit[6]
        incl_range = [incl + errors_fit[6][0], incl - errors_fit[6][1]]
        # dist = best_pars[7][0]
        # ebmv = best_pars[8][0]

    Mstar_range = [Mstar + errors_fit[0][0], Mstar - errors_fit[0][1]]

    W_range = [W + errors_fit[1][0], W - errors_fit[1][1]]

    tms_range = [tms + errors_fit[2][0], tms - errors_fit[2][1]]

    incl_range = [incl + errors_fit[3][0], incl - errors_fit[3][1]]

    oblat = W2oblat(W)
    ob_max, ob_min = W2oblat(W_range[0]), W2oblat(W_range[1])
    oblat_range = [ob_max, ob_min]
    # print(oblat_range, oblat)

    if tms <= 1.0:
        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr="014")
    else:
        Rpole, logL = geneva_interp(Mstar, oblat, tms, Zstr="014")
    # Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')

    Rpole_range = [0.0, 100.0]
    logL_range = [0.0, 100000.0]

    for mm in Mstar_range:
        for oo in oblat_range:
            for tt in tms_range:
                if tt <= 1.0:
                    Rpolet, logLt, _ = geneva_interp_fast(mm, oo, tt, Zstr="014")
                else:
                    Rpolet, logLt = geneva_interp(mm, oo, tt, Zstr="014")
                # Rpolet, logLt, _ = geneva_interp_fast(mm, oo, tt, Zstr='014')
                if Rpolet > Rpole_range[0]:
                    Rpole_range[0] = Rpolet
                    # print('Rpole max is now = {}'.format(Rpole_range[0]))
                if Rpolet < Rpole_range[1]:
                    Rpole_range[1] = Rpolet
                    # print('Rpole min is now = {}'.format(Rpole_range[1]))
                if logLt > logL_range[0]:
                    logL_range[0] = logLt
                    # print('logL max is now = {}'.format(logL_range[0]))
                if logLt < logL_range[1]:
                    logL_range[1] = logLt
                    # print('logL min is now = {}'.format(logL_range[1]))

    beta_range = [beta(oblat_range[0], is_ob=True), beta(oblat_range[1], is_ob=True)]

    beta_par = beta(oblat, is_ob=True)

    Req = oblat * Rpole
    Req_max, Req_min = oblat_range[0] * Rpole_range[0], oblat_range[1] * Rpole_range[1]

    omega = oblat2w(oblat)
    wcrit = np.sqrt(8.0 / 27.0 * G * Mstar * Msun / (Rpole * Rsun) ** 3)
    vsini = omega * wcrit * (Req * Rsun) * np.sin(np.deg2rad(incl)) * 1e-5

    A_roche = (
        4.0
        * np.pi
        * (Rpole * Rsun) ** 2
        * (
            1.0
            + 0.19444 * omega ** 2
            + 0.28053 * omega ** 4
            - 1.9014 * omega ** 6
            + 6.8298 * omega ** 8
            - 9.5002 * omega ** 10
            + 4.6631 * omega ** 12
        )
    )

    Teff = ((10.0 ** logL) * Lsun / sigma / A_roche) ** 0.25

    Teff_range = [0.0, 50000.0]
    vsini_range = [0.0, 10000.0]
    for mm in Mstar_range:
        for oo in oblat_range:
            for tt in tms_range:
                for ii in incl_range:
                    if tt <= 1.0:
                        rr, ll, _ = geneva_interp_fast(mm, oo, tt, Zstr="014")
                    else:
                        rr, ll = geneva_interp(mm, oo, tt, Zstr="014")
                    wcrit = np.sqrt(8.0 / 27.0 * G * mm * Msun / (rr * Rsun) ** 3)
                    # print(rr, oo)
                    omega_ = oblat2w(oo)
                    vsinit = (
                        omega_
                        * wcrit
                        * (oo * rr * Rsun)
                        * np.sin(np.deg2rad(ii))
                        * 1e-5
                    )
                    if vsinit > vsini_range[0]:
                        vsini_range[0] = vsinit
                        # print("vsini max is now = {}".format(vsini_range[0]))

                    if vsinit < vsini_range[1]:
                        vsini_range[1] = vsinit
                        # print("vsini min is now = {}".format(vsini_range[1]))
                    A_roche = (
                        4.0
                        * np.pi
                        * (rr * Rsun) ** 2
                        * (
                            1.0
                            + 0.19444 * omega_ ** 2
                            + 0.28053 * omega_ ** 4
                            - 1.9014 * omega_ ** 6
                            + 6.8298 * omega_ ** 8
                            - 9.5002 * omega_ ** 10
                            + 4.6631 * omega_ ** 12
                        )
                    )

                    Teff_ = ((10.0 ** ll) * Lsun / sigma / A_roche) ** 0.25
                    if Teff_ > Teff_range[0]:
                        Teff_range[0] = Teff_
                        # print('Teff max is now = {}'.format(Teff_range[0]))
                    if Teff_ < Teff_range[1]:
                        Teff_range[1] = Teff_
                        # print('Teff min is now = {}'.format(Teff_range[1]))

    file1.writelines(
        r"$R_{\rm eq}/R_{\rm p}$"
        + " & ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Derived  \\\ \n".format(
            oblat, oblat_range[0] - oblat, oblat - oblat_range[1]
        )
    )
    params_to_print.append(
        "Oblateness = {0:.3f} +{1:.3f} -{2:.3f}".format(
            oblat, oblat_range[0] - oblat, oblat - oblat_range[1]
        )
    )
    file1.writelines(
        r"$R_{\rm eq}\,[R_\odot]$"
        + " & ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Derived  \\\ \n".format(
            Req, Req_max - Req, Req - Req_min
        )
    )
    params_to_print.append(
        "Equatorial radius = {0:.3f} +{1:.3f} -{2:.3f}".format(
            Req, Req_max - Req, Req - Req_min
        )
    )
    file1.writelines(
        r"$\log(L)\,[L_\odot]$"
        + " & ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Derived  \\\ \n".format(
            logL, logL_range[0] - logL, logL - logL_range[1]
        )
    )
    params_to_print.append(
        "Log Luminosity  = {0:.3f} +{1:.3f} -{2:.3f}".format(
            logL, logL_range[0] - logL, logL - logL_range[1]
        )
    )
    file1.writelines(
        r"$\beta$"
        + " & ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Derived \\\ \n".format(
            beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]
        )
    )
    params_to_print.append(
        "Beta  = {0:.3f} +{1:.3f} -{2:.3f}".format(
            beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]
        )
    )
    file1.writelines(
        r"$v \sin i\,\rm[km/s]$"
        + " & ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Derived  \\\ \n".format(
            vsini, vsini_range[0] - vsini, vsini - vsini_range[1]
        )
    )
    params_to_print.append(
        "vsini = {0:.3f} +{1:.3f} -{2:.3f}".format(
            vsini, vsini_range[0] - vsini, vsini - vsini_range[1]
        )
    )
    file1.writelines(
        r"$T_{\rm eff}$"
        + " & ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Derived  \\\ \n".format(
            Teff, Teff_range[0] - Teff, Teff - Teff_range[1]
        )
    )
    params_to_print.append(
        "Teff = {0:.3f} +{1:.3f} -{2:.3f}".format(
            Teff, Teff_range[0] - Teff, Teff - Teff_range[1]
        )
    )

    L = ["\hline \n", "\end{tabular} \n" "\end{table} \n"]

    file1.writelines(L)

    file1.close()

    params_print = " \n".join(map(str, params_to_print))

    return params_to_print


def BVcolors(FOLDER_DEFS, LIM, params, minfo, listpar, dims, lbdarr, models):
    params[3] = np.cos(np.deg2rad(params[3]))
    dist = params[4]
    ebmv = params[5]
    mod = griddataBA(minfo, models, params[:-LIM], listpar, dims)  # erg/s/cm2/micron
    dista = 1e3 / dist
    norma = (10 / dista) ** 2
    mod = mod * norma
    flux_mod = pyasl.unred(lbdarr * 1e4, mod, ebv=-1 * ebmv, R_V=3.1)

    vega_path = FOLDER_DEFS + "vega.dat"
    vega = np.loadtxt(vega_path)
    wv_vg = vega[:, 0]  # will be in \AA
    fx_vg = vega[:, 1]  # erg s-1 cm-2 \AA-1
    fU, fB, fV, fR, fI = convolution_JohnsonsZP(wv_vg, fx_vg)  # zero points magnitudes.

    mU, mB, mV, mR, mI = convolution_Johnsons(
        lbdarr * 1e4, flux_mod * 1e-4, fU, fB, fV, fR, fI
    )
    # print(mU, mB, mV, mR, mI)

    return mU, mB, mV, mR, mI


def linfit(x, y, ssize=0.05, yerr=np.empty(0)):
    """
    linfit() - retorna um array (y) normalizado, em posicoes de x

    x eh importante, pois y pode ser nao igualmente amostrado.
    x e y devem estar em ordem crescente.

    ssize = % do tamanho de y; numero de pontos usados nas extremidades
    para a media do contínuo. 'ssize' de .5 à 0 (exclusive).

    OUTPUT: y, yerr (if given)

    .. code:: python

        #Example:
        import numpy as np
        import matplotlib.pyplot as plt
        import pyhdust.phc as phc
        import pyhdust.spectools as spt

        wv = np.linspace(6500, 6600, 101)
        flx = (np.arange(101)[::-1])/100.+1+phc.normgauss(4, x=wv,
        xc=6562.79)*5

        plt.plot(wv, flx)
        normflx = spt.linfit(wv, flx)
        plt.plot(wv, normflx, ls='--')

        plt.xlabel(r'$\lambda$ ($\AA$)')
        plt.ylabel('Flux (arb. unit)')

    .. image:: _static/spt_linfit.png
        :align: center
        :width: 500
    """
    ny = np.array(y)[:]
    if ssize < 0 or ssize > 0.5:
        # _warn.warn("Invalid ssize value...", stacklevel=2)
        ssize = 0
    ssize = int(ssize * len(y))
    if ssize == 0:
        ssize = 1
    medx0, medx1 = np.average(x[:ssize]), np.average(x[-ssize:])
    if ssize > 9:
        medy0, medy1 = np.median(ny[:ssize]), np.median(ny[-ssize:])
    else:
        medy0, medy1 = np.average(ny[:ssize]), np.average(ny[-ssize:])
    new_y = medy0 + (medy1 - medy0) * (x - medx0) / (medx1 - medx0)
    idx = np.where(new_y != 0)
    ny[idx] = ny[idx] / new_y[idx]
    if len(yerr) == 0.0:
        return ny
    else:
        yerr = yerr / np.average(new_y)
        return ny, yerr


def lineProf(x, flx, lbc, flxerr=np.empty(0), hwidth=1000.0, ssize=0.05):
    """
    lineProf() - retorna um array (flx) normalizado e um array x em
    VELOCIDADES. `lbc` deve fornecido em mesma unidade de x para conversão
    lambda -> vel. Se vetor x jah esta em vel., usar funcao linfit().

    x eh importante, pois y pode ser nao igualmente amostrado.
    x e y devem estar em ordem crescente.

    ssize = % do tamanho de y; numero de pontos usados nas extremidades
    para a media do contínuo. 'ssize' de .5 à 0 (exclusive).

    OUTPUT: vel (array), flx (array)
    """
    x = (x - lbc) / lbc * c * 1e-5  # km/s
    idx = np.where(np.abs(x) <= 1.001 * hwidth)
    if len(flxerr) == 0:
        flux = linfit(x[idx], flx[idx], ssize=ssize)  # yerr=flxerr,
        if len(x[idx]) == 0:
            print("Wrong `lbc` in the lineProf function")
        return x[idx], flux
    else:
        flux, flxerr = linfit(x[idx], flx[idx], yerr=flxerr[idx], ssize=ssize)
        if len(x[idx]) == 0:
            print("Wrong `lbc` in the lineProf function")
        return x[idx], flux, flxerr


def Sliding_Outlier_Removal(x, y, window_size, sigma=3.0, iterate=1):
    """
    Sigmaclip routine by Jonathan Labadie-Bartz
    """
    # remove NANs from the data
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # make sure that the arrays are in order according to the x-axis
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]

    # tells you the difference between the last and first x-value
    x_span = x.max() - x.min()
    i = 0
    x_final = x
    y_final = y
    while i < iterate:
        i += 1
        x = x_final
        y = y_final

        # empty arrays that I will append not-clipped data points to
        x_good_ = np.array([])
        y_good_ = np.array([])

        # Creates an array with all_entries = True. index where you want to remove outliers are set to False
        tf_ar = np.full((len(x),), True, dtype=bool)
        ar_of_index_of_bad_pts = np.array([])  # not used anymore

        # this is how many days (or rather, whatever units x is in) to slide the window center when finding the outliers
        slide_by = window_size / 5.0

        # calculates the total number of windows that will be evaluated
        Nbins = int((int(x.max() + 1) - int(x.min())) / slide_by)

        for j in range(Nbins + 1):
            # find the minimum time in this bin, and the maximum time in this bin
            x_bin_min = x.min() + j * (slide_by) - 0.5 * window_size
            x_bin_max = x.min() + j * (slide_by) + 0.5 * window_size

            # gives you just the data points in the window
            x_in_window = x[(x > x_bin_min) & (x < x_bin_max)]
            y_in_window = y[(x > x_bin_min) & (x < x_bin_max)]

            # if there are less than 5 points in the window, do not try to remove outliers.
            if len(y_in_window) > 5:

                # Removes a linear trend from the y-data that is in the window.
                y_detrended = detrend(y_in_window, type="linear")
                y_in_window = y_detrended
                # print(np.median(m_in_window_))
                y_med = np.median(y_in_window)

                # finds the Median Absolute Deviation of the y-pts in the window
                y_MAD = MAD(y_in_window)

                # This mask returns the not-clipped data points.
                # Maybe it is better to only keep track of the data points that should be clipped...
                mask_a = (y_in_window < y_med + y_MAD * sigma) & (
                    y_in_window > y_med - y_MAD * sigma
                )
                # print(str(np.sum(mask_a)) + '   :   ' + str(len(m_in_window)))
                y_good = y_in_window[mask_a]
                x_good = x_in_window[mask_a]

                y_bad = y_in_window[~mask_a]
                x_bad = x_in_window[~mask_a]

                # keep track of the index --IN THE ORIGINAL FULL DATA ARRAY-- of pts to be clipped out
                try:
                    clipped_index = np.where([x == z for z in x_bad])[1]
                    tf_ar[clipped_index] = False
                    ar_of_index_of_bad_pts = np.concatenate(
                        [ar_of_index_of_bad_pts, clipped_index]
                    )
                except IndexError:
                    # print('no data between {0} - {1}'.format(x_in_window.min(), x_in_window.max()))
                    pass

        ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)

        x_final = x[tf_ar]
        y_final = y[tf_ar]
    return (x_final, y_final)


def fwhm2sigma(fwhm):
    """Function to convert FWHM to standard deviation (sigma)"""
    return fwhm * np.sqrt(2.0) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)


def gauss(x, *p):
    height, mu, fwhm = p
    return height * np.exp(-((x - mu) ** 2.0) / (2.0 * fwhm2sigma(fwhm) ** 2))


def delta_v(vel, flux, line):
    """
    Calcula o shift na velocidade, baseado no centro ajustado p gaussiana
    """
    # coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    v_corte = np.zeros(2)
    # if line == "Ha":
    #    i_corte = np.argsort(np.abs((flux - 1.0) - (flux.max() - 1.0) * 0.5))
    # else:
    i_corte = np.argsort(np.abs((flux - 1.0) - (flux.min() - 1.0) * 0.5))

    v_corte[0] = vel[i_corte[0]]

    v_aux = vel[i_corte[1]]
    count = 2
    while v_corte[0] * v_aux > 0:
        v_aux = vel[i_corte[count]]
        count += 1
    v_corte[1] = v_aux
    v_corte.sort()
    delt_v = 10.0

    v_trechos = np.array(
        [v_corte[0] - delt_v, v_corte[0], v_corte[1], v_corte[1] + delt_v]
    )

    # asas = np.where(
    #     (vel > v_trechos[0]) * (vel < v_trechos[1])
    #     + (vel > v_trechos[2]) * (vel < v_trechos[3])
    # )

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [flux.max() - 1.0, v_trechos.mean(), v_trechos[2] - v_trechos[1]]
    coeff, var_matrix = curve_fit(gauss, vel, flux - 1.0, p0=p0)

    return coeff[1]
