# __init__

from .utilities import (
    kde_scipy,
    set_ranges,
    find_lim,
    bin_data,
    jy2cgs,
    find_nearest,
    find_neighbours,
    geneva_interp,
    geneva_interp_fast,
    griddataBA,
    griddataBAtlas,
    print_to_latex,
    BVcolors,
    linfit,
    lineProf,
)
from .models_data import (
    model_reader,
    read_stellar_prior,
    read_iue,
    read_votable,
    combine_sed,
    read_opd_pol,
    read_observables,
)

# from .stats import lnlike, lnprob, lnprior, emcee_inference
from .plots import residuals, residuals_line, residuals_POL, traceplot
from .hpd import hpd_grid
from .be_theory import w2wbig, wbig2w, oblat2w, W2oblat, obl2W, hfrac2tms, beta

# from .star_info import star_information
from .corner_HDR import corner, hist2d, quantile

__version__ = "1.0"
__all__ = (
    "kde_scipy",
    "set_ranges",
    "find_lim",
    "bin_data",
    "jy2cgs",
    "find_nearest",
    "find_neighbours",
    "geneva_interp",
    "geneva_interp_fast",
    "griddataBA",
    "griddataBAtlas",
    "print_to_latex",
    "BVcolors",
    "linfit",
    "lineProf",
    "model_reader",
    "read_stellar_prior",
    "read_iue",
    "read_votable",
    "read_opd_pol",
    "combine_sed",
    "read_observables",
    # "lnlike",
    # "lnprob",
    # "lnprior",
    # "emcee_inferences",
    "residuals",
    "residuals_line",
    "residuals_POL",
    "traceplot",
    "hpd_grid",
    "w2wbig",
    "wbig2w",
    "oblat2w",
    "W2oblat",
    "obl2W",
    "hfrac2tms",
    "beta",
    # "star_information",
    "corner",
    "hist2d",
    "quantile",
)
