def circularD(dates):
    """
    Compute the average date of occurrence and DOYs after Bloeschl et al. 2017 (https://www.science.org/doi/10.1126/science.aan2506)
    """
    doys = []
    ms = []
    for date in dates:
        doy, m = get_doy_nyears(pd.Timestamp(date)) 
        doys.append(doy)
        ms.append(m)
    ## Convert day of occurance into angular value
    thetas = np.array(doys) * np.pi / np.array(ms)
    n = len(dates)  # number of years
    X = 1/n * np.sum(np.cos(thetas))
    Y = 1/n * np.sum(np.sin(thetas))
    m = 1/n * np.sum(np.array(ms))
    R = np.sqrt(X**2 + Y**2)
    if X>0 and Y>0:
        D = np.arctan(Y/X) * m/(np.pi*2)
    if X<0:
        D = np.arctan(Y/X) * m/(np.pi*2)+np.pi
    if X>0 and Y<0:
        D = np.arctan(Y/X) * m/(np.pi*2)+2*np.pi
    return D, doys

def circular(dates):
    """
    Compute modified theil sen slope after Bloeschl et al. 2017 (https://www.science.org/doi/10.1126/science.aan2506).
    Return median slope, and 0.1, 0.9 percentiles of slope.
    """
    doys = []
    ms = []
    for date in dates:
        doy, m = get_doy_nyears(pd.Timestamp(date)) 
        doys.append(doy)
        ms.append(m)
    alpha=0.9
    x = np.arange(0, len(doys))
    y = np.array(doys)
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    k = np.zeros(deltay.shape)
    
    mask1 = deltay > 365.25/2
    mask2 = deltay < -365.25/2
    
    k[mask1] = -365.25
    k[mask2] = 365.25
    slopes = (deltay[deltax > 0] + k[deltax > 0])  / deltax[deltax > 0]
    slopes.sort()
    medslope = np.median(slopes)
    
    medinter = np.median(y) - medslope * np.median(x)
    # Now compute confidence intervals
    if alpha > 0.5:
        alpha = 1. - alpha
    
    z = distributions.norm.ppf(alpha / 2.)
    # This implements (2.6) from Sen (1968)
    _, nxreps = _find_repeats(x)
    _, nyreps = _find_repeats(y)
    nt = len(slopes)       # N in Sen (1968)
    ny = len(y)            # n in Sen (1968)
        # Equation 2.6 in Sen (1968):
    sigsq = 1/18. * (ny * (ny-1) * (2*ny+5) -
                         sum(k * (k-1) * (2*k + 5) for k in nxreps) -
                         sum(k * (k-1) * (2*k + 5) for k in nyreps))
        # Find the confidence interval indices in `slopes`
    try:
        sigma = np.sqrt(sigsq)
        Ru = min(int(np.round((nt - z*sigma)/2.)), len(slopes)-1)
        Rl = max(int(np.round((nt + z*sigma)/2.)) - 1, 0)
        delta = slopes[[Rl, Ru]]
    except (ValueError, IndexError):
        delta = (np.nan, np.nan)
    return medslope, delta[0], delta[1]