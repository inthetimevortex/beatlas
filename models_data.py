import numpy as np
from glob import glob
from astropy.io import fits
from beatlas.utilities import (
    bin_data,
    jy2cgs,
    kde_scipy,
    lineProf,
    Sliding_Outlier_Removal,
    delta_v,
)
from astropy.io.votable import parse_single_table
from scipy.interpolate import griddata
import pandas as pd
import emcee
from icecream import ic
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

lines_dict = {"Ha": 6562.801, "Hb": 4861.363, "Hd": 4101.74, "Hg": 4340.462}


def model_reader(fname, HALPHA, HBETA, HDELTA, HGAMMA, POL):
    """
    Reads npz file with the models
    """
    data = np.load(fname + ".npz", allow_pickle=True)
    minfo = data["minfo"]
    listpar = data["listpar"]
    dims = data["dims"]
    if not POL:
        models_SED = data["models_SED"]
        lbd_SED = data["lbd_SED"]
        models_combined = [models_SED]
        lbd_combined = [lbd_SED]
        try:
            models_combined.append(data["models_Ha"])
            lbd_combined.append(data["lbd_Ha"])
        except:
            pass

        if HALPHA:
            # models_combined.append(data["models_Ha"])
            # lbd_combined.append(data["lbd_Ha"])
            # EWs = data["EWs"]
            # FWHMs = data["FWHMs"]
            EWs = []
            FWHMs = []
        else:
            EWs = []
            FWHMs = []

        if HBETA:
            models_combined.append(data["models_Hb"])
            lbd_combined.append(data["lbd_Hb"])
        if HDELTA:
            models_combined.append(data["models_Hd"])
            lbd_combined.append(data["lbd_Hd"])
        if HGAMMA:
            models_combined.append(data["models_Hg"])
            lbd_combined.append(data["lbd_Hg"])
    else:
        models_combined = [data["models_POL"]]
        lbd_combined = [data["lbd_POL"]]
        EWs = []
        FWHMs = []

    return minfo, models_combined, lbd_combined, listpar, dims, EWs, FWHMs


def read_stellar_prior(FOLDER_FIGS, STAR, RANGES, filename):
    # chain = np.load(FOLDER_FIGS + STAR + "/" + filename, allow_pickle=True)
    # Ndim = np.shape(chain)[-1]
    # flatchain = chain.reshape((-1, Ndim))
    reader = emcee.backends.HDFBackend(FOLDER_FIGS + STAR + "/" + filename)
    try:
        tau = reader.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except:
        burnin = int(2 * 20000.0)
        thin = int(0.5 * 20000.0)

    flatchain = reader.get_chain(discard=burnin, flat=True, thin=thin)

    mas = flatchain[:, 0]
    obl = flatchain[:, 1]
    age = flatchain[:, 2]

    range_mas = np.linspace(np.min(RANGES[0]), np.max(RANGES[0]), 200)
    range_obl = np.linspace(np.min(RANGES[1]), np.max(RANGES[1]), 200)
    range_age = np.linspace(np.min(RANGES[2]), np.max(RANGES[2]), 200)

    pdf_mas = kde_scipy(x=mas, x_grid=range_mas, bandwidth=0.005)
    pdf_obl = kde_scipy(x=obl, x_grid=range_obl, bandwidth=0.005)
    pdf_age = kde_scipy(x=age, x_grid=range_age, bandwidth=0.005)

    range_priors = [range_mas, range_obl, range_age]
    pdf_priors = [pdf_mas, pdf_obl, pdf_age]
    # eps = 1e-8
    # pdf_mas[np.abs(pdf_mas) < eps] = 0
    # pdf_obl[np.abs(pdf_obl) < eps] = 0
    # pdf_age[np.abs(pdf_age) < eps] = 0



    # ic(pdf_mas)
    # ic(range_mas)
    plt.plot(range_mas, pdf_mas)
    plt.savefig('mass.png')
    plt.close()    
    plt.plot(range_obl, pdf_obl)
    plt.savefig('obl.png')
    plt.close()
    plt.plot(range_age, pdf_age)
    plt.savefig('age.png')
    plt.close()

    # plt.show()

    return range_priors, pdf_priors


def read_iue(FOLDER_DATA, STAR, lbdarr):
    iue_list = glob(FOLDER_DATA + STAR + "/UV/*")

    if type(iue_list) is list:
        file_name = np.copy(iue_list)
    else:
        file_name = str(iue_list)

    fluxes, waves, errors = [], [], []

    if file_name[0][-3:] == "csv":
        file_iue = FOLDER_DATA + STAR + "/" + str(file_name)
        wave, flux, sigma = np.loadtxt(str(file_iue), delimiter=",").T
        fluxes = np.concatenate((fluxes, flux * 1e4), axis=0)
        waves = np.concatenate((waves, wave * 1e-4), axis=0)
        errors = np.concatenate((errors, sigma * 1e4), axis=0)

    else:
        # Combines the observations from all files in the folder, taking the good quality ones
        for fname in file_name:
            # print(file_iue)
            with fits.open(fname) as hdulist:
                tbdata = hdulist[1].data
                wave = tbdata.field("WAVELENGTH") * 1e-4  # mum
                flux = tbdata.field("FLUX") * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum
                sigma = tbdata.field("SIGMA") * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum

                # Filter of bad data: '0' is good data
                qualy = tbdata.field("QUALITY")
                idx = np.where((qualy == 0))
                wave = wave[idx]
                sigma = sigma[idx]
                flux = flux[idx]

                idx = np.where((flux > 0.0))
                wave = wave[idx]
                sigma = sigma[idx]
                flux = flux[idx]

                fluxes = np.concatenate((fluxes, flux), axis=0)
                waves = np.concatenate((waves, wave), axis=0)
                errors = np.concatenate((errors, sigma), axis=0)

    wave_lim_min_iue = min(waves)
    wave_lim_max_iue = 0.290
    indx = np.where(((waves >= wave_lim_min_iue) & (waves <= wave_lim_max_iue)))
    waves, fluxes, errors = waves[indx], fluxes[indx], errors[indx]

    # sort the combined observations in all files
    new_wave, new_flux, new_sigma = zip(*sorted(zip(waves, fluxes, errors)))

    nbins = 200
    xbin, ybin, dybin = bin_data(new_wave, new_flux, nbins, exclude_empty=True)

    # just to make sure that everythingm is in order
    ordem = xbin.argsort()
    wave = xbin[ordem]
    flux = ybin[ordem]
    sigma = dybin[ordem]

    for i in range(len(sigma)):
        if sigma[i] < flux[i] * 0.01:
            sigma[i] = flux[i] * 0.01

    return wave, flux, sigma


def read_votable(FOLDER_DATA, STAR):
    vo_file = glob(FOLDER_DATA + STAR + "/*.xml")[0]

    try:
        tb1 = parse_single_table(vo_file, verify="ignore")
        t1 = tb1.array
        wave = t1["Wavelength"][:]  # Angstrom
        flux = t1["Flux"][:]  # erg/cm2/s/A
        sigma = t1["Error"][:]  # erg/cm2/s/A
    except RuntimeWarning:
        tb1 = parse_single_table(vo_file, verify="ignore")
        t1 = tb1.array
        wave = t1["SpectralAxis0"][:]  # Angstrom
        flux = t1["Flux0"][:]  # erg/cm2/s/A
        sigma = [0.0] * len(flux)  # erg/cm2/s/A

    new_wave, new_flux, new_sigma = zip(*sorted(zip(wave, flux, sigma)))

    new_wave = list(new_wave)
    new_flux = list(new_flux)
    new_sigma = list(new_sigma)

    wave = np.copy(new_wave) * 1e-4
    flux = np.copy(new_flux) * 1e4
    sigma = np.copy(new_sigma) * 1e4

    for i in range(len(sigma)):
        if sigma[i] != 0:
            if sigma[i] < flux[i] * 0.1:
                sigma[i] = flux[i] * 0.1

    keep = wave > 0.34
    wave, flux, sigma = wave[keep], flux[keep], sigma[keep]

    if STAR == "HD37795":
        fname = FOLDER_DATA + STAR + "/alfCol.txt"
        data = np.loadtxt(
            fname,
            dtype={
                "names": ("lbd", "flux", "dflux", "source"),
                "formats": (np.float, np.float, np.float, "|S20"),
            },
        )
        wave = np.hstack([wave, data["lbd"]])
        flux = np.hstack([flux, jy2cgs(1e-3 * data["flux"], data["lbd"])])
        sigma = np.hstack([sigma, jy2cgs(1e-3 * data["dflux"], data["lbd"])])

    if STAR == "HD58715":
        fname = FOLDER_DATA + STAR + "/bcmi_radio.dat"
        data = np.loadtxt(
            fname,
            dtype={
                "names": ("lbd", "flux", "dflux", "source"),
                "formats": (np.float, np.float, np.float, "|S20"),
            },
        )
        wave = np.hstack([wave, data["lbd"]])
        flux = np.hstack([flux, jy2cgs(data["flux"], data["lbd"])])
        sigma = np.hstack([sigma, jy2cgs(data["dflux"], data["lbd"])])

    return wave, flux, sigma


def combine_sed(LBD_RANGE, wave, flux, sigma, models, lbd):
    """
    Combines SED parts into 1 array
    """
    if LBD_RANGE == "UV":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 0.3  # mum
    if LBD_RANGE == "UV+VIS":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 0.8  # mum
    if LBD_RANGE == "UV+VIS+NIR":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 5.0  # mum
    if LBD_RANGE == "UV+VIS+NIR+MIR":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 40.0  # mum
    if LBD_RANGE == "UV+VIS+NIR+MIR+FIR":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 350.0  # mum
    if LBD_RANGE == "UV+VIS+NIR+MIR+FIR+MICROW+RADIO" or LBD_RANGE == "FULLSED":
        wave_lim_min = 0.13  # mum
        wave_lim_max = np.max(wave)  # mum
    if LBD_RANGE == "VIS+NIR+MIR+FIR+MICROW+RADIO":
        wave_lim_min = 0.39  # mum
        wave_lim_max = np.max(wave)  # mum
    if LBD_RANGE == "NIR+MIR+FIR+MICROW+RADIO":
        wave_lim_min = 0.7  # mum
        wave_lim_max = np.max(wave)  # mum
    if LBD_RANGE == "MIR+FIR+MICROW+RADIO":
        wave_lim_min = 5.0  # mum
        wave_lim_max = np.max(wave)  # mum
    if LBD_RANGE == "FIR+MICROW+RADIO":
        wave_lim_min = 40.0  # mum
        wave_lim_max = np.max(wave)  # mum
    if LBD_RANGE == "MICROW+RADIO":
        wave_lim_min = 1e3  # mum
        wave_lim_max = np.max(wave)  # mum
    if LBD_RANGE == "RADIO":
        wave_lim_min = 1e6  # mum
        wave_lim_max = np.max(wave)  # mum

    ordem = wave.argsort()
    wave = wave[ordem]
    flux = flux[ordem]
    sigma = sigma[ordem]

    idx = np.where((wave >= wave_lim_min) & (wave <= wave_lim_max))
    wave = wave[idx]
    flux = flux[idx]
    sigma = sigma[idx]
    models_new = np.zeros([len(models), len(wave)])

    models[models <= 0] = 1e-20
    for i in range(len(models)):  # A interpolacao
        models_new[i, :] = griddata(
            np.log10(lbd), np.log10(models[i]), np.log10(wave), method="linear"
        )

    # to log space
    logF_data = np.log10(flux)
    dlogF_data = sigma / flux
    logF_grid = models_new

    return wave, logF_data, dlogF_data, logF_grid


def read_opd_pol(FOLDER_DATA, STAR):
    """
    Reads polarization data_pos

    Usage:
    wave, flux, sigma = read_opd_pol

    """

    table_csv = glob(FOLDER_DATA + STAR + "/" + "POL/*iscor.csv")[0]

    # Reading opd data from csv (beacon site)
    if table_csv != "hpol.npy":
        csv_file = table_csv
        df = pd.read_csv(csv_file)
        JD = df["#MJD"] + 2400000
        Filter = df["filt"]
        # flag = df['flag']
        lbd = np.array([0.3656, 0.4353, 0.5477, 0.6349, 0.8797])
        P = np.array(df["P"])  # * 100  # Units of percentage
        SIGMA = np.array(df["sigP"])

        # Plot P vs lambda
        Pol, error, wave = [], [], []
        nu, nb, nv, nr, ni = 0.0, 0.0, 0.0, 0.0, 0.0
        wu, wb, wv, wr, wi = 0.0, 0.0, 0.0, 0.0, 0.0
        eu, eb, ev, er, ei = 0.0, 0.0, 0.0, 0.0, 0.0

        filtros = np.unique(Filter)
        for h in range(len(JD)):
            for filt in filtros:
                if Filter[h] == filt:  # and flag[h] is not 'W':
                    if filt == "u":
                        wave.append(lbd[0])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wu = wu + P[h]  # * 100.
                        nu = nu + 1.0
                        eu = eu + (SIGMA[h]) ** 2.0
                    if filt == "b":
                        wave.append(lbd[1])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wb = wb + P[h]  # * 100.
                        nb = nb + 1.0
                        eb = eb + (SIGMA[h]) ** 2.0
                    if filt == "v":
                        wave.append(lbd[2])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wv = wv + P[h]  # * 100.
                        nv = nv + 1.0
                        ev = ev + (SIGMA[h]) ** 2.0
                    if filt == "r":
                        wave.append(lbd[3])
                        Pol.append(P[h])  # 100. * P[i])
                        error.append(SIGMA[h])
                        wr = wr + P[h]  # * 100.
                        nr = nr + 1.0
                        er = er + (SIGMA[h]) ** 2.0
                    if filt == "i":
                        wave.append(lbd[4])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wi = wi + P[h]  # * 100.
                        ni = ni + 1.0
                        ei = ei + (SIGMA[h]) ** 2.0
        try:
            eu = np.sqrt(eu / nu)
            eb = np.sqrt(eb / nb)
            ev = np.sqrt(ev / nv)
            er = np.sqrt(er / nr)
            ei = np.sqrt(ei / ni)
        except RuntimeWarning:
            # eu = np.sqrt(eu / nu)
            eb = np.sqrt(eb / nb)
            ev = np.sqrt(ev / nv)
            er = np.sqrt(er / nr)
            ei = np.sqrt(ei / ni)

        try:
            sigma = np.array([eu, eb, ev, er, ei])  # * 100
        except RuntimeWarning:
            sigma = np.array([eb, ev, er, ei])  # * 100

        try:
            mean_u = wu / nu
        except RuntimeWarning:
            mean_u = 0.0
            print("u sem dado")

        try:
            mean_b = wb / nb
        except RuntimeWarning:
            mean_b = 0.0
            print("b sem dado")

        try:
            mean_v = wv / nv
        except RuntimeWarning:
            mean_v = 0.0
            print("v sem dado")

        try:
            mean_r = wr / nr
        except RuntimeWarning:
            mean_r = 0.0
            print("r sem dado")

        try:
            mean_i = wi / ni
        except RuntimeWarning:
            mean_i = 0.0
            print("i sem dado")

        try:
            flux = np.array([mean_u, mean_b, mean_v, mean_r, mean_i])
        except RuntimeWarning:
            flux = np.array([mean_b, mean_v, mean_r, mean_i])
        # flux = np.array([mean_b, mean_v, mean_r, mean_i])
        wave = np.copy(lbd)
    else:
        # print(table_csv, star, folder_data)
        wave, flux, sigma = np.load(FOLDER_DATA + STAR + "/" + str(table_csv))

    return wave, flux, sigma


def read_line_spectra(FOLDER_DATA, STAR, lbd, models, linename):
    file_name = glob(FOLDER_DATA + STAR + "/SPECTRA/*" + linename + "*.npz")[0]
    line_data = np.load(file_name)
    wave = line_data['lbd']
    flux = line_data['flux']
    # file_name = glob(FOLDER_DATA + STAR + "/SPECTRA/*" + linename + "*.fits")[0]
    # hdulist = fits.open(file_name)
    # fits_header = hdulist[0].header
    # flux = hdulist[0].data
    # wave = fits_header["CRVAL1"] + fits_header["CDELT1"] * np.arange(len(flux))

    cs = 299792.458  # km/s
    lbd_central = lines_dict[linename]

    # gives the line profiles in velocity space
    vl, fx = lineProf(wave, flux, hwidth=5000.0, lbc=lbd_central)
    vel, flux = Sliding_Outlier_Removal(vl, fx, 30, 3, 15)

    radv = delta_v(vel, flux, linename)
    print("RADIAL VELOCITY = {0}".format(radv))
    vel = vel - radv
    wl = cs * lbd_central / (cs - vel)

    wave = wl * 1e-4  # mum
    interpolator = interp1d(wave, flux)
    # lbdarr = lbd[1:-1]
    flux = interpolator(lbd)
    # print(lineProf(lbd, flux, hwidth=5000.0, lbc=lbd_central * 1e-4))
    sigma = np.zeros(len(flux))
    sigma.fill(0.017656218)

    remove_partHa = False
    if remove_partHa:
        lbd1 = lbd[:203]
        lbd2 = lbd[244:]
        lbd = np.append(lbd1, lbd2)
        novo_models = np.zeros([len(models), len(lbd)])
        flux_new1 = flux[:203]
        flux_new2 = flux[244:]
        flux_new = np.append(flux_new1, flux_new2)
        sigma_new = np.append(sigma[:203], sigma[244:])
        for i in range(len(models)):
            novo_models[i] = np.append(models[i][:203], models[i][244:])
    # novo_models = np.zeros((len(models), len(lbdarr)))
    # for i in range(len(models)):
    #     mm = models[i][1:-1][np.isfinite(fluxes)]
    #     novo_models[i] = mm

    return flux_new, sigma_new, lbd, novo_models


def read_observables(SED, POL, HALPHA, HBETA, HDELTA, HGAMMA, LBD_RANGE, FOLDER_DATA, STAR, lbd, models):

    # data_flux = []
    # data_wave = []
    # data_sigma = []
    # grid_flux = []

    if SED:
        index = 0

        if "UV" in LBD_RANGE or LBD_RANGE == "FULLSED":
            wave0, flux0, sigma0 = read_iue(FOLDER_DATA, STAR, lbd[index])
        else:
            wave0, flux0, sigma0 = [], [], []
        if LBD_RANGE != "UV":
            wave1, flux1, sigma1 = read_votable(FOLDER_DATA, STAR)
            if STAR == "gammaCas":
                wave1, flux1, sigma1 = np.loadtxt(FOLDER_DATA + STAR + "/vband.dat").T
        else:
            wave1, flux1, sigma1 = [], [], []

        wave = np.hstack([wave0, wave1])
        flux = np.hstack([flux0, flux1])
        sigma = np.hstack([sigma0, sigma1])
        # print(wave, flux, sigma)

        data_wave, data_flux, data_sigma, grid_flux = combine_sed(
            LBD_RANGE, wave, flux, sigma, models[index], lbd[index]
        )
    else:
        data_wave, data_flux, data_sigma, grid_flux = [], [], [], []

    if HALPHA:
        index = 1
        linename = "Ha"
        fluxHa, sigmaHa, lbdHa, modelsHa = read_line_spectra(
            FOLDER_DATA, STAR, lbd[index], models[index], linename
        )
    else:
        fluxHa, sigmaHa, lbdHa, modelsHa = [], [], [], []
    
    if HBETA:
        index = 2
        linename = "Hb"
        fluxHb, sigmaHb, lbdHb, modelsHb = read_line_spectra(
            FOLDER_DATA, STAR, lbd[index], models[index], linename
        )
    else:
        fluxHb, sigmaHb, lbdHb, modelsHb = [], [], [], []
    
    if HDELTA:
        index = 3
        linename = "Hd"
        fluxHd, sigmaHd, lbdHd, modelsHd = read_line_spectra(
            FOLDER_DATA, STAR, lbd[index], models[index], linename
        )
    else:
        fluxHd, sigmaHd, lbdHd, modelsHd = [], [], [], []
    
    if HGAMMA:
        index = 4
        linename = "Hg"
        fluxHg, sigmaHg, lbdHg, modelsHg = read_line_spectra(
            FOLDER_DATA, STAR, lbd[index], models[index], linename
        )
    else:
        fluxHg, sigmaHg, lbdHg, modelsHg = [], [], [], []


    if HALPHA or HBETA or HDELTA or HGAMMA:
        data_wave = [data_wave, lbdHa, lbdHb, lbdHd, lbdHg]
        data_flux = [data_flux, fluxHa, fluxHb, fluxHd, fluxHg]
        data_sigma = [data_sigma, sigmaHa, sigmaHb, sigmaHd, sigmaHg]
        grid_flux = [grid_flux, modelsHa, modelsHb, modelsHd, modelsHg]

    if POL:
        index = 0
        data_wave, data_flux, data_sigma = read_opd_pol(FOLDER_DATA, STAR)

        return data_wave, data_flux, data_sigma, models[index]
    else:
        return data_wave, data_flux, data_sigma, grid_flux

