from beatlas.star_info import fname, star_information
import os
import random

FOLDER_DATA = "../data/"
FOLDER_FIGS = "../figures/"
FOLDER_DEFS = "../defs/"
FOLDER_TABLES = "../tables/"
FOLDER_MODELS = "../models/"

(
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
    STEPS,
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
) = star_information(fname)

if os.path.isdir(FOLDER_FIGS + STAR) is False:
    os.mkdir(FOLDER_FIGS + STAR)

if LBD_RANGE.lower() != "pol":
    SED = True
    POL = False
else:
    SED = False
    POL = True

if CORNER_COLOR == "random" or CORNER_COLOR == "":
    CORNER_COLOR = random.choice(
        [
            "blue",
            "dark blue",
            "teal",
            "green",
            "yellow",
            "orange",
            "red",
            "purple",
            "violet",
            "pink",
        ]
    )

if CORNER_COLOR == "blue":
    COLOR = "xkcd:cornflower"
    COLOR_HIST = "xkcd:powder blue"
    COLOR_DENS = "xkcd:clear blue"

elif CORNER_COLOR == "dark blue":
    COLOR = "xkcd:petrol"
    COLOR_HIST = "xkcd:cool blue"
    COLOR_DENS = "xkcd:ocean"

elif CORNER_COLOR == "teal":
    COLOR = "xkcd:dark sea green"
    COLOR_HIST = "xkcd:aqua marine"
    COLOR_DENS = "xkcd:seafoam blue"

elif CORNER_COLOR == "green":
    COLOR = "xkcd:forest green"
    COLOR_HIST = "xkcd:light grey green"
    COLOR_DENS = "xkcd:grass green"

elif CORNER_COLOR == "yellow":
    COLOR = "xkcd:sandstone"
    COLOR_HIST = "xkcd:pale gold"
    COLOR_DENS = "xkcd:sunflower"

elif CORNER_COLOR == "orange":
    COLOR = "xkcd:cinnamon"
    COLOR_HIST = "xkcd:light peach"
    COLOR_DENS = "xkcd:bright orange"

elif CORNER_COLOR == "red":
    COLOR = "xkcd:deep red"
    COLOR_HIST = "xkcd:salmon"
    COLOR_DENS = "xkcd:reddish"

elif CORNER_COLOR == "purple":
    COLOR = "xkcd:medium purple"
    COLOR_HIST = "xkcd:soft purple"
    COLOR_DENS = "xkcd:plum purple"

elif CORNER_COLOR == "violet":
    COLOR = "xkcd:purpley"
    COLOR_HIST = "xkcd:pale violet"
    COLOR_DENS = "xkcd:blue violet"

elif CORNER_COLOR == "pink":
    COLOR = "xkcd:pinky"
    COLOR_HIST = "xkcd:rosa"
    COLOR_DENS = "xkcd:pink red"


if MODEL == "2022 BeAtlas":
    LABELS = [
        r"$M\,[M_\odot]$",
        r"$W$",
        r"$t/t_\mathrm{ms}$",
        r"$i[\mathrm{^o}]$",
        r"$\pi\,[mas]$",
        r"E(B-V)",
    ]
    LABELS2 = LABELS

elif MODEL == "acol" or MODEL == "pol" or MODEL == "aara":
    LABELS = [
        r"$M\,[\mathrm{M_\odot}]$",
        r"$W$",
        r"$t/t_\mathrm{ms}$",
        r"$\log \, n_0 \, [\mathrm{cm^{-3}}]$",
        r"$R_\mathrm{D}\, [R_\star]$",
        r"$n$",
        r"$i[\mathrm{^o}]$",
        r"$\pi\,[\mathrm{mas}]$",
        r"E(B-V)",
    ]
    LABELS2 = [
        r"$M$",
        r"$W$",
        r"$t/t_\mathrm{ms}$",
        r"$\log \, n_0 $",
        r"$R_\mathrm{D}$",
        r"$n$",
        r"$i$",
        r"$\pi$",
        r"E(B-V)",
    ]


elif MODEL == "old disk BeAtlas":
    LABELS = [
        r"$M\,[\mathrm{M_\odot}]$",
        r"$W$",
        r"$\Sigma_0 \, [\mathrm{g/cm^{-2}}]$",
        r"$n$",
        r"$i[\mathrm{^o}]$",
        r"$\pi\,[\mathrm{mas}]$",
        r"E(B-V)",
    ]
    LABELS2 = [
        r"$M$",
        r"$W$",
        r"$\\Sigma_0 $",
        r"$R_\mathrm{D}$",
        r"$n$",
        r"$i$",
        r"$\pi$",
        r"E(B-V)",
    ]

if POL:
    LABELS = LABELS[:-2]
    LABELS2 = LABELS2[:-2]
if INCLUDE_RV:
    LABELS = LABELS + [r"$R_\mathrm{V}$"]
    LABELS2 = LABELS2 + [r"$R_\mathrm{V}$"]
