import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyAstronomy import pyasl
from beatlas.utilities import griddataBA, griddataBAtlas, lineProf, linfit
from icecream import ic

sns.set_style("ticks")
lines_dict = {"Ha": 0.6562801, "Hb": 4861.363, "Hd": 4101.74, "Hg": 4340.462}


def traceplot(NDIM, LABELS, sampler, fname):
    """
    Makes and saves the trace plot

    """

    fig, axes = plt.subplots(NDIM, figsize=(7, 12), sharex=True)

    # Load the chain
    samples = sampler.get_chain()

    for i in range(NDIM):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(LABELS[i])  # , fontsize=16)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number")  # , fontsize=16)

    plt.savefig(fname + ".png", bbox_inches="tight")


def residuals(
    MODEL,
    LBD_RANGE,
    INCLUDE_RV,
    LIM,
    STEPS,
    HALPHA,
    sampler,
    data_wave,
    data_flux,
    data_sigma,
    grid_flux,
    minfo,
    listpar,
    dims,
    burnin,
    thin,
):
    """
    Create residuals plot separated from the corner
    For the SED and the lines

    """
    sns.set_style("ticks")
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    par_list = []
    inds = np.random.randint(len(flat_samples), size=300)
    for ind in inds:
        params = flat_samples[ind]
        par_list.append(params)

    uplim = data_sigma == 0.0
    keep = np.logical_not(uplim)
    logF_list = np.zeros([len(par_list), len(data_flux)])

    for i, params in enumerate(par_list):
        if MODEL != "BeAtlas2015_disk":
            logF_list[i] = griddataBA(minfo, grid_flux, params[:-LIM], listpar, dims)
        else:
            logF_list[i] = griddataBAtlas(
                minfo, grid_flux, params[:-LIM], listpar, dims, isig=dims["sig0"],
            )

    keep = data_sigma != 0.0
    data_flux_notlog = 10 ** data_flux
    data_sigma_notlog = data_sigma * data_flux_notlog
    if MODEL == "acol" or MODEL == "aara" or MODEL == "BeAtlas2022_disk":
        bottom, left = 0.80, 0.51  # 0.80, 0.48  # 0.75, 0.48
        width, height = 0.96 - left, 0.97 - bottom
        ax1 = plt.axes([left, bottom, width, height])
        ax2 = plt.axes([left, bottom - 0.095, width, 0.075])
        ax1.get_xaxis().set_visible(False)
        ms = 6
    else:
        if HALPHA:
            bottom, left = 0.87, 0.61  # 0.80, 0.48  # 0.75, 0.48
            width, height = 0.97 - left, 0.97 - bottom
            ax1 = plt.axes([left, bottom, width, height])
            ax2 = plt.axes([left, bottom - 0.06, width, 0.05])
        else:
            bottom, left = 0.85, 0.58  # 0.80, 0.48  # 0.75, 0.48
            width, height = 0.96 - left, 0.97 - bottom
            ax1 = plt.axes([left, bottom, width, height])
            ax2 = plt.axes([left, bottom - 0.08, width, 0.07])
        ax1.get_xaxis().set_visible(False)
        ms = 2

    # Plot Models
    for i in range(len(par_list)):
        if MODEL == "BeAtlas2022_phot":
            dist_tmp = par_list[i][4]
            ebv_tmp = par_list[i][5]
            if INCLUDE_RV:
                rv_tmp = par_list[6][0]
            else:
                rv_tmp = 3.1
        elif MODEL == "BeAtlas2015_disk":
            dist_tmp = par_list[i][5]
            ebv_tmp = par_list[i][6]
            if INCLUDE_RV:
                rv_tmp = par_list[7][0]
            else:
                rv_tmp = 3.1
        else:
            dist_tmp = par_list[i][7]
            ebv_tmp = par_list[i][8]
            if INCLUDE_RV:
                rv_tmp = par_list[9][0]
            else:
                rv_tmp = 3.1
        F_temp = pyasl.unred(
            data_wave * 1e4, 10 ** logF_list[i], ebv=-1 * ebv_tmp, R_V=rv_tmp
        )
        dist = 1e3 / dist_tmp
        norma = (10.0 / dist) ** 2
        F_temp = F_temp * norma
        ax1.plot(data_wave, F_temp, color="gray", alpha=0.1, lw=0.6)

        ax2.plot(
            data_wave[keep],
            (data_flux_notlog[keep] - F_temp[keep]) / data_sigma_notlog[keep],
            "ks",
            # ms=int(ms - 2),
            ms=ms,
            alpha=0.1,
        )

    # Plot Data
    ax1.errorbar(
        data_wave[keep],
        data_flux_notlog[keep],
        yerr=data_sigma_notlog[keep],
        ls="",
        marker="o",
        alpha=0.5,
        ms=ms,
        color="k",
        linewidth=1,
    )
    arrow = u"$\u2193$"
    antikeep = np.invert(keep)
    ax1.errorbar(
        data_wave[antikeep],
        data_flux_notlog[antikeep],
        yerr=data_sigma_notlog[antikeep],
        ls="",
        marker=arrow,
        # alpha=0.5,
        ms=int(ms + 7),
        color="k",
        linewidth=1,
    )
    ax2.axhline(y=0.0, ls=(0, (5, 10)), lw=0.6, color="k")
    ax2.set_xlabel(r"$\lambda\,\mathrm{[\mu m]}$", fontsize=14)
    ax1.set_ylabel(
        r"$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}\, \mu m^{-1}]}$", fontsize=14,
    )
    ax2.set_ylabel(r"$(F-F_\mathrm{m})/\sigma$", fontsize=14)
    ax2.sharex(ax1)
    if LBD_RANGE != "UV":
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(-7, 0))

    return


def residuals_line(
    linename,
    MODEL,
    LBD_RANGE,
    INCLUDE_RV,
    LIM,
    STEPS,
    SED,
    sampler,
    data_wave_line,
    data_flux_line,
    data_sigma_line,
    grid_flux_line,
    minfo,
    listpar,
    dims,
    burnin,
    thin,
):
    """
    Create residuals plot separated from the corner
    For the SED and the lines

    """
    lbd_central = lines_dict[linename]
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    par_list = []
    inds = np.random.randint(len(flat_samples), size=300)
    for ind in inds:
        params = flat_samples[ind]
        par_list.append(params)

    F_list = np.zeros([len(par_list), len(data_flux_line)])
    for i, params in enumerate(par_list):
        F_list_unnorm = griddataBA(minfo, grid_flux_line, params[:-LIM], listpar, dims,)
        F_list[i] = linfit(data_wave_line, F_list_unnorm)

    if not SED:
        bottom, left = 0.85, 0.58  # 0.80, 0.48  # 0.75, 0.48
        width, height = 0.96 - left, 0.97 - bottom
        ax1 = plt.axes([left, bottom, width, height])
        ax2 = plt.axes([left, bottom - 0.08, width, 0.07])
    else:
        bottom, left = 0.64, 0.61  # 0.80, 0.48  # 0.75, 0.48
        width, height = 0.97 - left, 0.74 - bottom
        ax1 = plt.axes([left, bottom, width, height])
        ax2 = plt.axes([left, bottom - 0.06, width, 0.05])
    
    ax1.get_xaxis().set_visible(False)
    ms = 2

    vl, fxx = lineProf(data_wave_line, data_flux_line, hwidth=3000.0, lbc=lbd_central)
    for i in range(len(par_list)):
        vl, fx = lineProf(data_wave_line, F_list[i], hwidth=3000.0, lbc=lbd_central)
        # ax1.plot(lbd_line, F_list[i], color='gray', alpha=0.1)
        ax1.plot(vl, fx, color="gray", alpha=0.1)
        ax2.plot(
            vl, (fxx - fx) / data_sigma_line[1:], "ks", ms=ms, alpha=0.1
        )

    ax1.errorbar(
        vl,
        fxx,
        yerr=data_sigma_line[1:],
        ls="",
        marker="o",
        alpha=0.5,
        ms=ms,
        color="k",
        linewidth=1,
    )

    ax1.set_ylabel("Norm. Flux", fontsize=14)
    # ax1.set_xlim(min(vl), max(vl))
    ax1.set_xlim(-2000, +2000)
    ax2.set_xlim(-2000, +2000)
    # ax1.legend(loc='lower right')
    # ax1.set_title(line)
    # Residuals
    # ax2.plot(lbd_line, (flux_line - F_list[-1])/dflux_line, marker='o', color='k', alpha=0.5)


    ax2.set_ylabel(r"$(F-F_\mathrm{m})/\sigma$", fontsize=14)
    ax2.axhline(y=0.0, ls=(0, (5, 10)), lw=0.7, color="k")
    # ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$')#, fontsize=14)
    ax2.set_xlabel("Velocity [km/s]", fontsize=14)

    return


def residuals_POL(
    MODEL,
    LIM,
    STEPS,
    sampler,
    data_wave,
    data_flux,
    data_sigma,
    grid_flux,
    minfo,
    listpar,
    dims,
    burnin,
    thin,
):
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    par_list = []
    inds = np.random.randint(len(flat_samples), size=300)
    for ind in inds:
        params = flat_samples[ind]
        par_list.append(params)

    F_list = np.zeros([len(par_list), len(data_flux)])

    bottom, left = 0.82, 0.51  # 0.80, 0.48  # 0.75, 0.48
    width, height = 0.96 - left, 0.97 - bottom
    ax1 = plt.axes([left, bottom, width, height])
    ax2 = plt.axes([left, bottom - 0.08, width, 0.065])
    ax1.get_xaxis().set_visible(False)
    ms = 6

    for i, params in enumerate(par_list):
        F_list[i] = griddataBA(minfo, grid_flux, params[:-LIM], listpar, dims)
        ax1.plot(data_wave, F_list[i], color="gray", alpha=0.1, lw=0.6)

    ax2.plot(data_wave, (data_flux - F_list[-1]) / data_sigma, "ks", ms=ms, alpha=0.2)
    ax1.errorbar(
        data_wave,
        data_flux,
        yerr=data_sigma,
        ls="",
        marker="o",
        alpha=0.5,
        ms=ms,
        color="k",
        linewidth=1,
    )

    ax1.set_xlim(0.34, 0.9)
    ax2.axhline(y=0.0, ls=(0, (5, 10)), lw=0.7, color="k")
    ax2.set_xlabel("$\lambda\,\mathrm{[\mu m]}$", fontsize=16)
    ax1.set_ylabel(r"$P_{\%}$", fontsize=16)
    ax2.sharex(ax1)
    ax2.set_ylabel("$(P-P_\mathrm{m})/\sigma$", fontsize=16)

    return
