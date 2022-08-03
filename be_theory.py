import numpy as np
from icecream import ic


# WBIG2W
def wbig2w(W, oblat=False):
    """
    Converts W=vrot/vorb into w=Omega/Omega_c

    Usage:
    w(or oblat) = wbig2w(W, oblat=False(or True))
    """
    # from stuff import oblat2w
    # from constants import pi

    tolerance = 1e-3
    if W < tolerance:
        w = 0.0
    elif 1.0 - W < tolerance:
        w = 1.0
    else:
        w0 = tolerance
        w1 = 1.0 - tolerance
        delt = 1e-5
        res = 1.0
        while res > delt:
            w = 0.5 * (w1 + w0)
            gam = 2.0 * np.cos((np.pi + np.arccos(w)) / 3.0)
            W_half = np.sqrt(gam ** 3 / w)
            if W_half < W:
                w0 = w
            else:
                w1 = w
            res = np.abs((w1 - w0) / w0)

    if oblat:
        w = oblat2w(w)

    return w


# W2WBIG
def w2wbig(w, oblat=False):
    """
    Converts w=Omega/Omega_c into W=vrot/vorb

    Usage:
    W = w2wbig(w, oblat=False)
    """
    # from stuff import oblat2w
    # from constants import pi

    if oblat:
        w = oblat2w(w)

    gam = 2.0 * np.cos((np.pi + np.arccos(w)) / 3.0)
    W = np.sqrt(gam ** 3 / w)

    return W


# ==============================================================================
def oblat2w(oblat):
    """
    Author: Rodrigo Vieira
    Converts oblateness into wc=Omega/Omega_crit
    Ekstrom et al. 2008, Eq. 9

    Usage:
    w = oblat2w(oblat)
    """
    if (np.min(oblat) < 1.0) or (np.max(oblat) > 1.5):
        print("Warning: values out of allowed range")

    oblat = np.array([oblat]).reshape((-1))
    nw = len(oblat)
    w = np.zeros(nw)

    for iw in range(nw):
        if oblat[iw] <= 1.0:
            w[iw] = 0.0
        elif oblat[iw] >= 1.5:
            w[iw] = 1.0
        else:
            w[iw] = (1.5 ** 1.5) * np.sqrt(2.0 * (oblat[iw] - 1.0) / oblat[iw] ** 3.0)

    if nw == 1:
        w = w[0]

    return w


# ==============================================================================
def W2oblat(W):

    oblat = W ** 2 / 2 + 1

    return oblat


# ==============================================================================
def obl2W(oblat):

    W = np.sqrt(2 * (oblat - 1))

    return W


# ==============================================================================
def hfrac2tms(Hfrac, inverse=False):
    """
    Converts nuclear hydrogen fraction into fractional time in
    the main-sequence, (and vice-versa) based on the polynomial
    fit of the average of this relation for all B spectral types
    and rotational velocities.

    Usage:
    t = hfrac2tms(Hfrac, inverse=False)
    or
    Hfrac = hfrac2tms(t, inverse=True)
    """
    if not inverse:
        coef = np.array([-0.57245754, -0.8041484, -0.51897195, 1.00130795])
        tms = coef.dot(np.array([Hfrac ** 3, Hfrac ** 2, Hfrac ** 1, Hfrac ** 0]))
    else:
        # interchanged parameter names
        coef = np.array([-0.74740597, 0.98208541, -0.64318363, -0.29771094, 0.71507214])
        tms = coef.dot(
            np.array([Hfrac ** 4, Hfrac ** 3, Hfrac ** 2, Hfrac ** 1, Hfrac ** 0])
        )

    # solving problem at lower extreme
    if type(tms) is list or type(tms) is np.ndarray:
        tms[tms < 0.0] = 0.0
    else:
        if tms < 0.0:
            tms = 0.0

    return tms


def beta(par, is_ob=False):
    r""" Calculate the :math:`\beta` value from Espinosa-Lara for a given
    rotation rate :math:`w_{\rm frac} = \Omega/\Omega_c`

    If ``is_ob == True``, it consider the param as ob (instead of
    :math:`w_{\rm frac}`). """

    # Ekstrom et al. 2008, Eq. 9
    if is_ob:
        wfrac = (1.5 ** 1.5) * np.sqrt(2.0 * (par - 1.0) / par ** 3)
    else:
        wfrac = par

    # avoid exceptions
    if wfrac == 0:
        return 0.25
    elif wfrac == 1:
        return 0.13535
    elif wfrac < 0 or wfrac > 1:
        return 0.0

    # Espinosa-Lara VLTI-School 2013 lecture, slide 18...
    delt = 1.0
    omega1 = 0.0
    omega = wfrac
    while delt >= 1e-5:
        f = (3.0 / (2.0 + omega ** 2)) ** 3 * omega ** 2 - wfrac ** 2
        df = -108.0 * omega * (omega ** 2 - 1.0) / (omega ** 2 + 2.0) ** 4
        omega1 = omega - f / df
        delt = np.abs(omega1 - omega) / omega
        omega = omega1

    nthe = 100
    theta = np.linspace(0, np.pi / 2, nthe + 1)[1:]
    grav = np.zeros(nthe)
    teff = np.zeros(nthe)
    corr = np.zeros(nthe)
    beta = 0.0

    for ithe in range(nthe):

        delt = 1.0
        r1 = 0.0
        r = 1.0
        while delt >= 1e-5:
            f = (
                omega ** 2 * r ** 3 * np.sin(theta[ithe]) ** 2
                - (2.0 + omega ** 2) * r
                + 2.0
            )
            df = 3.0 * omega ** 2 * r ** 2 * np.sin(theta[ithe]) ** 2 - (
                2.0 + omega ** 2
            )
            r1 = r - f / df
            delt = np.abs(r1 - r) / r
            r = r1

        delt = 1.0
        n1 = 0.0
        ftheta = (
            1.0 / 3.0 * omega ** 2 * r ** 3 * np.cos(theta[ithe]) ** 3
            + np.cos(theta[ithe])
            + np.log(np.tan(theta[ithe] / 2.0))
        )
        n = theta[ithe]
        while delt >= 1e-5:
            f = np.cos(n) + np.log(np.tan(n / 2.0)) - ftheta
            df = -np.sin(n) + 1.0 / np.sin(n)
            if df > 0.0:
                n1 = n - f / df
            else:
                n1 = 0.0
            delt = abs(n1 - n) / n
            n = n1

        grav[ithe] = np.sqrt(
            1.0 / r ** 4
            + omega ** 4 * r ** 2 * np.sin(theta[ithe]) ** 2
            - 2.0 * omega ** 2 * np.sin(theta[ithe]) ** 2 / r
        )

        corr[ithe] = np.sqrt(np.tan(n) / np.tan(theta[ithe]))
        teff[ithe] = corr[ithe] * grav[ithe] ** 0.25

    u = ~np.isnan(teff)
    coef = np.polyfit(np.log(grav[u]), np.log(teff[u]), 1)
    beta = coef[0]

    return beta
