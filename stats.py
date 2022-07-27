import numpy as np
from PyAstronomy import pyasl
from scipy.special import erf
from beatlas.be_theory import hfrac2tms, oblat2w
from beatlas.utilities import (
    jy2cgs,
    geneva_interp,
    geneva_interp_fast,
    find_nearest,
    griddataBA,
    griddataBAtlas,
)
from beatlas.constants import *


def lnlike(
    params, MODEL, STAR, PLX, SED, POL, LBD_RANGE, INCLUDE_RV, wave, flux, sigma, mod
):

    """
    Returns the likelihood probability function (-0.5 * chi2).

    """

    if SED:
        if MODEL == "befavor" or MODEL == "BeAtlas2022_phot":
            dist = params[4]
            ebmv = params[5]
        if MODEL == "aara" or MODEL == "acol":
            dist = params[7]
            ebmv = params[8]
        if MODEL == "BeAtlas2015_disk":
            dist = params[5]
            ebmv = params[6]

        if INCLUDE_RV:
            RV = params[-1]
        else:
            RV = 3.1

        # print(logF_UV)
        dist = 1e3 / PLX
        norma = (10 / dist) ** 2
        uplim = sigma == 0.0

        keep = np.logical_not(uplim)

        mod += np.log10(norma)
        tmp_flux = 10 ** mod

        flux_mod = pyasl.unred(wave * 1e4, tmp_flux, ebv=-1 * ebmv, R_V=RV)
        logF_mod = np.log10(flux_mod)

        rms = [1e-3 * 0.1] * len(flux[uplim])
        rms = np.array(rms)

        upper_lim = jy2cgs(10 ** flux[uplim], wave[uplim], inverse=True)
        mod_upper = jy2cgs(10 ** logF_mod[uplim], wave[uplim], inverse=True)

        # a parte dos uplims não é em log!
        if "UV" in LBD_RANGE:
            # TIRANDO O LOG!!!!
            onlyUV = np.logical_and(wave > 0.13, wave < 0.29)
            chi2_onlyUV = np.sum(
                (
                    (10 ** flux[onlyUV] - flux_mod[onlyUV]) ** 2.0
                    / (10 ** flux[onlyUV] * sigma[onlyUV]) ** 2.0
                )
            )
            N_onlyUV = len(flux[onlyUV])
            chi2_onlyUV_red = chi2_onlyUV / N_onlyUV
        else:
            chi2_onlyUV_red = 0.0
            N_onlyUV = 0.0

        if LBD_RANGE != "UV":  # se UV não é o único!
            rest = np.logical_and(wave > 0.3, keep)
            chi2_rest = np.sum(((flux[rest] - logF_mod[rest]) ** 2.0 / (sigma) ** 2.0))
            N_rest = len(flux[rest])
            # ic(N_rest)
            chi2_rest_red = chi2_rest / N_rest
        else:
            chi2_rest_red = 0.0
            N_rest = 0.0

        # TESTES DOS UPLIMS
        if "RADIO" in LBD_RANGE:
            if STAR == "HD37795" or STAR == "HD158427":  # and not flag.Ha:
                # print("## using uplim chi2!! ##")
                chi2_uplim = -2.0 * np.sum(
                    np.log(
                        (np.pi / 2.0) ** 0.5
                        * rms
                        * (1.0 + erf(((upper_lim - mod_upper) / ((2 ** 0.5) * rms))))
                    )
                )
                N_uplim = len(rms)
                chi2_uplim_red = chi2_uplim / N_uplim
            else:
                N_uplim = 0.0
                chi2_uplim_red = 0.0
        else:
            N_uplim = 0.0
            chi2_uplim_red = 0.0

    if POL:
        chi2 = np.sum((flux - mod) ** 2 / (sigma) ** 2.0)
    else:
        chi2 = (chi2_onlyUV_red + chi2_rest_red + chi2_uplim_red) * (
            N_onlyUV + N_rest + N_uplim
        )

    if chi2 is np.nan:
        chi2 = np.inf

    return -0.5 * chi2


def lnprior(
    params,
    RANGES,
    MODEL,
    STAR,
    SED,
    HALPHA,
    POL,
    PLX_PRIOR,
    PLX,
    D_PLX,
    VSINI_PRIOR,
    VSINI,
    D_VSINI,
    INCL_PRIOR,
    INCL,
    D_INCL,
    STELLAR_KDE_PRIOR,
    range_priors,
    pdf_priors,
    logF_mod,
    EW_model,
    FWHM_model,
):
    """ Calculates the chi2 for the priors set in user_settings

    Usage:
    chi2_prior = lnprior(params)
    """

    if MODEL == "BeAtlas2022_phot":
        if SED:
            Mstar, W, tms, cosi, dist, ebv = (
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
            )
        else:
            Mstar, W, tms, cosi = params[0], params[1], params[2], params[3]
        oblat = 1 + 0.5 * (W ** 2)
    if MODEL == "befavor":
        Mstar, oblat, Hfrac, cosi, dist, ebv = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
        )
        tms = np.max(np.array([hfrac2tms(Hfrac), 0.0]))
    if MODEL == "aara" or MODEL == "acol" or MODEL == "pol":
        if SED:
            Mstar, oblat, Hfrac, cosi, dist, ebv = (
                params[0],
                params[1],
                params[2],
                params[6],
                params[7],
                params[8],
            )
        else:
            Mstar, oblat, Hfrac, cosi = params[0], params[1], params[2], params[6]
        tms = np.max(np.array([hfrac2tms(Hfrac), 0.0]))
    if MODEL == "BeAtlas2015_disk":
        Mstar, oblat, Hfrac, cosi, dist, ebv = (
            params[0],
            params[1],
            0.3,
            params[4],
            params[5],
            params[6],
        )
        tms = np.max(np.array([hfrac2tms(Hfrac), 0.0]))

    # Reading Stellar Priors
    if STELLAR_KDE_PRIOR:
        temp, idx_mas = find_nearest(range_priors[0], value=Mstar)
        temp, idx_obl = find_nearest(range_priors[1], value=oblat)
        temp, idx_age = find_nearest(range_priors[2], value=Hfrac)
        chi2_stellar_prior = (
            1.0 / pdf_priors[0][idx_mas]
            + 1.0 / pdf_priors[1][idx_obl]
            + 1.0 / pdf_priors[2][idx_age]
        )
    else:
        chi2_stellar_prior = 0.0

    # Vsini prior
    if VSINI_PRIOR:
        if tms <= 1.0:
            Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr="014")
        else:
            Rpole, logL = geneva_interp(Mstar, oblat, tms, Zstr="014")
        wcrit = np.sqrt(8.0 / 27.0 * G * Mstar * Msun / (Rpole * Rsun) ** 3)
        vsin = (
            oblat2w(oblat)
            * wcrit
            * (Rpole * Rsun * oblat)
            * np.sin(np.arccos(cosi))
            * 1e-5
        )

        chi2_vsi = ((VSINI - vsin) / D_VSINI) ** 2.0
    else:
        chi2_vsi = 0

    # Distance prior
    if SED:
        if PLX_PRIOR:
            chi2_dis = ((PLX - dist) / D_PLX) ** 2.0
        else:
            chi2_dis = 0
    else:
        chi2_dis = 0
        chi2_vsi = 0

    # Inclination prior
    if INCL_PRIOR:
        inc = np.arccos(cosi) * 180.0 / np.pi  # obtaining inclination from cosi
        chi2_incl = ((INCL - inc) / D_INCL) ** 2.0
    else:
        chi2_incl = 0.0

    if HALPHA:
        if STAR == "HD37795":
            EW_data = -27.09
            EW_err = 1.94 * 2
            FWHM_data = 237.36943
            FWHM_err = 237.36943 * 0.03
            chi2_ew = ((EW_data - EW_model) / EW_err) ** 2.0
            chi2_fwhm = ((FWHM_data - FWHM_model) / FWHM_err) ** 2
        elif STAR == "HD58715":
            EW_data = -15.18
            EW_err = 1.15 * 2
            FWHM_data = 272.71297
            FWHM_err = 272.71297 * 0.03
            chi2_ew = ((EW_data - EW_model) / EW_err) ** 2.0
            chi2_fwhm = ((FWHM_data - FWHM_model) / FWHM_err) ** 2
    else:
        chi2_ew = 0.0
        chi2_fwhm = 0.0

    chi2_prior = (
        chi2_vsi + chi2_dis + chi2_stellar_prior + chi2_incl + chi2_ew + chi2_fwhm
    )

    if chi2_prior is np.nan:
        chi2_prior = -np.inf

    return -0.5 * chi2_prior


def lnprob(
    params,
    MODEL,
    STAR,
    LBD_RANGE,
    INCLUDE_RV,
    SED,
    POL,
    HALPHA,
    LIM,
    EWs,
    FWHMs,
    RANGES,
    PLX_PRIOR,
    PLX,
    D_PLX,
    VSINI_PRIOR,
    VSINI,
    D_VSINI,
    INCL_PRIOR,
    INCL,
    D_INCL,
    STELLAR_KDE_PRIOR,
    range_priors,
    pdf_priors,
    data_wave,
    data_flux,
    data_sigma,
    grid_flux,
    minfo,
    listpar,
    dims,
):
    """
    Calculates lnprob (lnprior + lnlike)

    """

    count = 0
    inside_ranges = True
    while inside_ranges * (count < len(params)):
        inside_ranges = (params[count] >= RANGES[count, 0]) * (
            params[count] <= RANGES[count, 1]
        )
        count += 1

    if inside_ranges:
        if SED:
            # index = 0
            if MODEL != "BeAtlas2015_disk":
                mod = griddataBA(minfo, grid_flux, params[:-LIM], listpar, dims,)
            else:
                mod = griddataBAtlas(
                    minfo, grid_flux, params[:-LIM], listpar, dims, isig=dims["sig0"],
                )

        if HALPHA:
            EW_model = griddataBA(minfo, EWs, params[:-LIM], listpar, dims,)
            FWHM_model = griddataBA(minfo, FWHMs, params[:-LIM], listpar, dims,)
        else:
            EW_model = 0.0
            FWHM_model = 0.0

        if POL:
            mod = griddataBA(minfo, grid_flux, params, listpar, dims)

        lp = lnprior(
            params,
            RANGES,
            MODEL,
            STAR,
            SED,
            HALPHA,
            POL,
            PLX_PRIOR,
            PLX,
            D_PLX,
            VSINI_PRIOR,
            VSINI,
            D_VSINI,
            INCL_PRIOR,
            INCL,
            D_INCL,
            STELLAR_KDE_PRIOR,
            range_priors,
            pdf_priors,
            mod,
            EW_model,
            FWHM_model,
        )

        lk = lnlike(
            params,
            MODEL,
            STAR,
            PLX,
            SED,
            POL,
            LBD_RANGE,
            INCLUDE_RV,
            data_wave,
            data_flux,
            data_sigma,
            mod,
        )

        lpost = lp + lk

        if not np.isfinite(lpost):
            return -np.inf
        else:
            return lpost
    else:
        return -np.inf
