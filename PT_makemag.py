import pyhdust as hdt
import numpy as np

# function for calculating the integrated flux given a filter bandpass;
# it uses pyhdust.doFilterConv()


def convolution_JohnsonsZP(wave, flux, zeropt=True):

    fU = hdt.doFilterConv(wave, flux, "U", zeropt=zeropt)
    fB = hdt.doFilterConv(wave, flux, "B", zeropt=zeropt)
    fV = hdt.doFilterConv(wave, flux, "V", zeropt=zeropt)
    fI = hdt.doFilterConv(wave, flux, "I", zeropt=zeropt)
    fR = hdt.doFilterConv(wave, flux, "R", zeropt=zeropt)

    return fU, fB, fV, fR, fI


def convolution_Johnsons(wave, flux, fU, fB, fV, fR, fI, zeropt=True):

    # print(hdt.doFilterConv(wave, flux, "U", zeropt=zeropt))

    mU = 2.5 * np.log10(fU / hdt.doFilterConv(wave, flux, "U", zeropt=zeropt))
    mB = 2.5 * np.log10(fB / hdt.doFilterConv(wave, flux, "B", zeropt=zeropt))
    mV = 2.5 * np.log10(fV / hdt.doFilterConv(wave, flux, "V", zeropt=zeropt))
    mR = 2.5 * np.log10(fR / hdt.doFilterConv(wave, flux, "R", zeropt=zeropt))
    mI = 2.5 * np.log10(fI / hdt.doFilterConv(wave, flux, "I", zeropt=zeropt))

    return mU, mB, mV, mR, mI


# Usage

# Load Vega's spectrum from pyhdust; your source can change, but remember which
# units you are using

# vega_path = os.path.join(hdt.hdtpath(), "refs/stars/", "vega.dat")
# vega = np.loadtxt(vega_path)
# wv_vg = vega[:, 0]  # will be in \AA
# fx_vg = vega[:, 1]  # erg s-1 cm-2 \AA-1
# fU, fB, fV, fR, fI = convolution_JohnsonsZP(wv_vg, fx_vg)  # zero points magnitudes.

# here, you use things from BeAtlas

# mU, mB, mV, mR, mI = convolution_Johnsons(lbdarr * 1e4, mod, fU, fB, fV, fR, fI)
