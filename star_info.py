import numpy as np
import sys

fname = sys.argv[1]


def star_information(fname):
    direc = "input/"

    values = np.loadtxt(direc + fname + ".txt", dtype=str)

    STAR = values[0][-1]
    PLX = float(values[1][-1])
    D_PLX = float(values[2][-1])
    VSINI = float(values[3][-1])
    D_VSINI = float(values[4][-1])
    INCL = float(values[5][-1])
    D_INCL = float(values[6][-1])
    MODEL = values[7][-1]
    A_PAR = float(values[8][-1])
    INCLUDE_RV = values[9][-1] == "True"
    WALKERS = int(values[10][-1])
    MAX_STEPS = int(values[11][-1])
    LBD_RANGE = values[12][-1]
    HALPHA = values[13][-1] == "True"
    HBETA = values[14][-1] == "True"
    HDELTA = values[15][-1] == "True"
    HGAMMA = values[16][-1] == "True"
    VSINI_PRIOR = values[17][-1] == "True"
    PLX_PRIOR = values[18][-1] == "True"
    INCL_PRIOR = values[19][-1] == "True"
    STELLAR_KDE_PRIOR = values[20][-1] == "True"
    PARALLEL = values[21][-1] == "True"
    PROCESSORS = int(values[22][-1])
    CORNER_COLOR = values[23][-1]

    return (
        STAR,
        PLX,
        D_PLX,
        VSINI,
        D_VSINI,
        INCL,
        D_INCL,
        MODEL,
        A_PAR,
        INCLUDE_RV,
        WALKERS,
        MAX_STEPS,
        LBD_RANGE,
        HALPHA,
        HBETA,
        HDELTA,
        HGAMMA,
        VSINI_PRIOR,
        PLX_PRIOR,
        INCL_PRIOR,
        STELLAR_KDE_PRIOR,
        PARALLEL,
        PROCESSORS,
        CORNER_COLOR,
    )
