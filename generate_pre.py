# -*- coding: utf-8 -*-
# -------------------------------

# @product：PyCharm


# -------------------------------

# @filename：generate_pre.py
# @teim：2025/11/1 10:24
# @email：2301110293@pku.edu.cn

# -------------------------------
import numpy as np

def AR1DFA(S, q=3, precision=0.01, alst=None, GMP=False, MP=False,pre_AR1DFA_dict=None):
    '''
    Calculate square of DFA fluctuation function of standard AR(1) process

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    alst : list or array_like,other Iterable eta
    Select special a list for grid search.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    pre_AR1DFA_dict : dict
    Prepared dictionary.

    Returns
    -------
    tuple
    (
    a : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(1) process)

    Fs2 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(1) process
    )
    '''
    if pre_AR1DFA_dict is None:
        if alst is None:
            a = np.arange(precision, 1, precision)
        else:
            a = np.array(alst)
        if GMP == True and MP == True:
            import warnings
            warnings.warn(
                "Multiple Precision package GMP（GNU Multiple Precision Arithmetic Library:gmpy2） and MP（mpmath）are conflict,default selection is GMP(Cpython)",
                SyntaxWarning)
        else:
            pass
        if GMP == True:
            import gmpy2
            gmpy2.get_context().precision = int(1 / precision)
            a = a / gmpy2.mpfr(1)
            S = S / gmpy2.mpfr(1)
        elif MP == True:
            import mpmath as mp
            mp.mp.prec = int(1 / precision)
            a = a / mp.mpf(1)
            S = S / mp.mpf(1)

        else:
            pass

        aa = np.expand_dims(a, axis=-1)
        consta = np.ones_like(aa)

        SS = np.expand_dims(S, axis=0)
        constS = np.ones_like(SS)

        aa = np.matmul(aa, constS)
        SS = np.matmul(consta, SS)

        if q == 1:

            Fs2 = (aa ** (2 * SS) * (((15 * aa ** 4 * (aa - 1) ** 3) * SS ** 3)
                                     + ((-60 * aa ** 4 * (aa - 1) ** 2 * (aa + 1)) * SS ** 2)
                                     + ((15 * aa ** 4 * (aa ** 3 + 5 * aa ** 2 + 5 * aa + 1)) * SS)
                                     + (-30 * aa ** 4 * (aa ** 3 + 5 * aa ** 2 + 5 * aa + 1)))
                   + (aa ** SS) * (((-60 * aa ** 2 * (aa - 1) ** 3 * (aa + 1) ** 2) * SS ** 2)
                                   + ((-180 * aa ** 2 * (aa - 1) * (aa + 1) ** 2) * SS)
                                   + (60 * aa ** 2 * (aa + 1) ** 2 * (aa ** 3 + 3 * aa ** 2 + 2)))
                   + ((((aa - 1) ** 5 * (aa + 1) ** 2) * SS ** 5)
                      + ((15 * aa * (aa - 1) ** 4 * (aa + 1)) * SS ** 4)
                      + ((-5 * (aa - 1) ** 3 * (4 * aa ** 4 - 6 * aa ** 3 - 14 * aa ** 2 - 6 * aa + 1)) * SS ** 3)
                      + ((-15 * (aa - 1) ** 2 * aa * (4 * aa ** 4 + 5 * aa ** 3 - 9 * aa ** 2 - 9 * aa + 1)) * SS ** 2)
                      + ((
                                 -71 * aa ** 7 - 177 * aa ** 6 + 169 * aa ** 5 + 155 * aa ** 4 - 80 * aa ** 3 - 34 * aa ** 2 + 42 * aa - 4) * SS)
                      + (-30 * aa ** 2 * (aa + 2) ** 2 * (aa ** 3 + aa ** 2 + aa + 1)))) \
                  / ((15 * (aa - 1) ** 7 * (aa + 1) * (SS ** 4 - SS ** 2)))

        elif q == 2:
            pass

        elif q == 3:
            J = 504 * SS ** 6 * (aa - 1) ** 6 * aa ** 2. - 7560 * SS ** 5 * (aa - 1) ** 5 * aa ** 2 * (
                    1 + aa) + 63 * SS ** 4 * (aa - 1) ** 4 * aa ** 2 * (
                        728 * aa ** 2 + 1664 * aa + 728) - 4 * SS ** 3 * (aa - 1) ** 3 * aa ** 2 * (aa + 1) * (
                        35910 * aa ** 2 + 102060 * aa + 35910) + 252 * SS ** 2 * (aa - 1) ** 2 * aa ** 2 * (
                        968 * aa ** 4 + 5428 * aa ** 3 + 9528 * aa ** 2 + 5428 * aa + 968) - 144 * SS ** 1 * (
                        aa ** 2 - 1) * aa ** 2 * (
                        1470 * aa ** 4 + 9450 * aa ** 3 + 22260 * aa ** 2 + 9450 * aa + 1470) + 72576 * aa ** 2 * (
                        aa ** 6 + 9 * aa ** 5 + 45 * aa ** 4 + 65 * aa ** 3 + 45 * aa ** 2 + 9 * aa + 1)
            K = 2 * SS ** 9 * (aa - 1) ** 9 * (aa + 1) + 63 * SS ** 8 * (aa - 1) ** 8 * aa - 12 * SS ** 7 * (
                    aa - 1) ** 7 * (5 * aa ** 3 - 26 * aa ** 2 - 26 * aa + 5) - 126 * SS ** 6 * (aa - 1) ** 6 * aa * (
                        7 * aa ** 2 - 30 * aa + 7) + 42 * SS ** 5 * (aa - 1) ** 5 * (aa + 1) * (
                        13 * aa ** 4 - 136 * aa ** 3 + 246 * aa ** 2 - 136 * aa + 13) + 63 * SS ** 4 * (
                        aa - 1) ** 4 * aa * (
                        49 * aa ** 4 - 644 * aa ** 3 + 710 * aa ** 2 - 644 * aa + 49) - 4 * SS ** 3 * (aa - 1) ** 3 * (
                        aa + 1) * (
                        410 * aa ** 6 - 5547 * aa ** 5 + 18498 * aa ** 4 - 26722 * aa ** 3 + 18498 * aa ** 2 - 5547 * aa + 410) - 252 * SS ** 2 * (
                        aa - 1) ** 2 * aa * (
                        9 * aa ** 6 - 446 * aa ** 5 + 143 * aa ** 4 - 2292 * aa ** 3 + 143 * aa ** 2 - 446 * aa + 9) + 144 * SS ** 1 * (
                        aa ** 2 - 1) * (
                        8 * aa ** 8 - 127 * aa ** 7 + 602 * aa ** 6 - 1393 * aa ** 5 + 1820 * aa ** 4 - 1393 * aa ** 3 + 602 * aa ** 2 - 127 * aa + 8) - 72576 * aa ** 2 * (
                        aa ** 6 + 9 * aa ** 5 + 45 * aa ** 4 + 65 * aa ** 3 + 45 * aa ** 2 + 9 * aa + 1)
            Fs2 = (J * aa ** SS + K) / (-63 * (aa - 1) ** 10 * SS ** 2 * (SS ** 6 - 14 * SS ** 4 + 49 * SS ** 2 - 36))
        else:
            pass
        a = a.astype(np.float64)
        Fs2p_AR1 = Fs2.astype(np.float64)
    elif pre_AR1DFA_dict is True:
        tempdict=get_pre_AR1DFA_dict(S)
        a = tempdict['a']
        Fs2p_AR1 = tempdict['Fs2p_AR1']
    elif isinstance(pre_AR1DFA_dict,str) is True:
        tempdict = get_pre_AR1DFA_dict(S,path=pre_AR1DFA_dict)
        a = tempdict['a']
        Fs2p_AR1 = tempdict['Fs2p_AR1']
    else:
        a=pre_AR1DFA_dict['a']
        Fs2p_AR1=pre_AR1DFA_dict['Fs2p_AR1']
    return a, Fs2p_AR1

def generate_pre_AR1DFA_dict(S,q=3,alst=None,precision=0.01, GMP=True,MP=False,save_nc=False,**kwargs):
    '''
    Calculate dict: square of DFA fluctuation function of standard AR(1) process

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    save_nc : bool
    Saved dict to netCDF4 format in local dir.

    Returns
    -------
    dict
    {
    a : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(1) process)

    Fs2p_AR1 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(1) process
    }
    '''
    import xarray as xr
    a, Fs2p_AR1 = AR1DFA(S=S, q=q,alst=alst, precision=precision, GMP=GMP,MP=MP)
    if save_nc is True:
        nc_name=kwargs.pop('nc_name',"Fs2p_AR1.nc")
        ds = xr.Dataset(
            {
                "a": (["ar1index"], np.array(a)),
                "Fs2p_AR1": (["ar1index", "S"], np.array(Fs2p_AR1)),
            },
            coords={
                "ar1index": np.arange(len(a)),
                "S": S,
            },
            attrs=dict(description="Fs2p_AR1 data.")
        )
        ds['a'].attrs = {"units": '1', 'long_name': "first-order autoregressive coefficient for AR(1) grid search"}
        ds.to_netcdf(nc_name)
        ds.close()
    else:
        pass
    pre_AR1DFA_dict = {'a': a,
                       'Fs2p_AR1': Fs2p_AR1}
    return pre_AR1DFA_dict
def ar1dfa3(s, a, ai=0):
    '''
    This function calculates the theoretical DFA-3 result of an AR(1) process
    i.e. the fluctuation function F(s)

    Parameters
    ----------
    s : numpy.ndarray
    all points for which a result is desired.

    a : float
    a is the real part of the AR(1) parameter.

    ai : float
    ai is the imaginary part.

    Returns
    -------

    '''

    '''s = array(s) * 1.0
    if (ai != 0):
        a = complex(a, ai)'''
    a=a+ai*1j
    """
    Both numerator and denominator are divided by s**8
    """

    J = 504 * s ** 6 * (a - 1) ** 6 * a ** 2 \
        - 7560 * s ** 5 * (a - 1) ** 5 * a ** 2 * (1 + a) \
        + 63 * s ** 4 * (a - 1) ** 4 * a ** 2 * (728 * a ** 2 + 1664 * a + 728) \
        - 4 * s ** 3 * (a - 1) ** 3 * a ** 2 * (a + 1) * (35910 * a ** 2 + 102060 * a + 35910) \
        + 252 * s ** 2 * (a - 1) ** 2 * a ** 2 * (968 * a ** 4 + 5428 * a ** 3 + 9528 * a ** 2 + 5428 * a + 968) \
        - 144 * s ** 1 * (a ** 2 - 1) * a ** 2 * (1470 * a ** 4 + 9450 * a ** 3 + 22260 * a ** 2 + 9450 * a + 1470) \
        + 72576 * a ** 2 * (a ** 6 + 9 * a ** 5 + 45 * a ** 4 + 65 * a ** 3 + 45 * a ** 2 + 9 * a + 1)
    K = 2 * s ** 9 * (a - 1) ** 9 * (a + 1) \
        + 63 * s ** 8 * (a - 1) ** 8 * a \
        - 12 * s ** 7 * (a - 1) ** 7 * (5 * a ** 3 - 26 * a ** 2 - 26 * a + 5) \
        - 126 * s ** 6 * (a - 1) ** 6 * a * (7 * a ** 2 - 30 * a + 7) \
        + 42 * s ** 5 * (a - 1) ** 5 * (a + 1) * (13 * a ** 4 - 136 * a ** 3 + 246 * a ** 2 - 136 * a + 13) \
        + 63 * s ** 4 * (a - 1) ** 4 * a * (49 * a ** 4 - 644 * a ** 3 + 710 * a ** 2 - 644 * a + 49) \
        - 4 * s ** 3 * (a - 1) ** 3 * (a + 1) * (
                    410 * a ** 6 - 5547 * a ** 5 + 18498 * a ** 4 - 26722 * a ** 3 + 18498 * a ** 2 - 5547 * a + 410) \
        - 252 * s ** 2 * (a - 1) ** 2 * a * (
                    9 * a ** 6 - 446 * a ** 5 + 143 * a ** 4 - 2292 * a ** 3 + 143 * a ** 2 - 446 * a + 9) \
        + 144 * s ** 1 * (a ** 2 - 1) * (
                    8 * a ** 8 - 127 * a ** 7 + 602 * a ** 6 - 1393 * a ** 5 + 1820 * a ** 4 - 1393 * a ** 3 + 602 * a ** 2 - 127 * a + 8) \
        - 72576 * a ** 2 * (a ** 6 + 9 * a ** 5 + 45 * a ** 4 + 65 * a ** 3 + 45 * a ** 2 + 9 * a + 1)
    Fs2 = (J * a ** s + K) / (-63 * (a - 1) ** 10 * s ** 2 * (s ** 6 - 14 * s ** 4 + 49 * s ** 2 - 36))
    return (Fs2)
def ar2dfa3(s, a1, a2):

    '''
    This function calculates the theoretical DFA-3 result of an AR(2) process,
    i.e. the fluctuation function F(s).

    Parameters
    ----------
    s : numpy.ndarray
    all points for which a result is desired.

    a1 : float
    a1 and a2 are the AR(2) parameters

    a2 : float
    a1 and a2 are the AR(2) parameters

    Returns
    -------
    Fs2 : numpy.ndarray
    The squre fluctuation function F(s)

    '''
    G1 = (((a1 + np.sqrt(a1 ** 2 + (4 + 0 * 1j) * a2)) / (-2 * a2)) ** (-1))
    G2 = (((a1 - np.sqrt(a1 ** 2 + (4 + 0 * 1j) * a2)) / (-2 * a2)) ** (-1))
    A1 = ((a1 / (1 - a2)) - G2) / (G1 - G2)
    A2 = 1 - A1
    Fs2 = ((A1 * ar1dfa3(s, G1.real, G1.imag) + A2 * ar1dfa3(s, G2.real, G2.imag)))
    return (abs(Fs2))

def ar2dfa3_MP(s, a1, a2):
    '''
    This function calculates the theoretical DFA-3 result of an AR(2) process,
    i.e. the fluctuation function F(s).
    based on MP（Multiple Precision Math:mpmath）

    Parameters
    ----------
    s : numpy.ndarray
    all points for which a result is desired.

    a1 : float
    a1 and a2 are the AR(2) parameters

    a2 : float
    a1 and a2 are the AR(2) parameters

    Returns
    -------
    Fs2 : numpy.ndarray
    The squre fluctuation function F(s)

    '''
    G1 = (((a1 + (a1 ** 2 + (4 + 0 * 1j) * a2)**0.5) / (-2 * a2)) ** (-1))
    G2 = (((a1 - (a1 ** 2 + (4 + 0 * 1j) * a2)**0.5) / (-2 * a2)) ** (-1))
    A1 = ((a1 / (1 - a2)) - G2) / (G1 - G2)
    A2 = 1 - A1
    Fs2 = ((A1 * ar1dfa3(s, G1.real, G1.imag) + A2 * ar1dfa3(s, G2.real, G2.imag)))
    return (abs(Fs2))

def AR2DFA_GMP(S,a1=None,a2=None,q=3,precision=0.01):
    '''
    The square of DFA fluctuation function of standard AR(1) process
    based on GMP（GNU Multiple Precision Arithmetic Library:gmpy2）

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a1 : NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    a2 : NoneType,list or array_like,other Iterable eta
    Select special a list for second-order autoregressive coefficient grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    tuple
    (
    aa1 : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process)

    aa2 : numpy.ndarray
    Grid of second-order autoregressive coefficient(AR(2) process)

    Fs2 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process
    )
    '''
    import gmpy2
    gmpy2.get_context().precision = int(1 / precision)
    if a1 is None:
        a1 = np.arange(-2 + precision, 2, precision)
    else:
        a1=np.array(a1)
    if a2 is None:
        a2 = np.arange(-1+precision,1,precision)
    else:
        a2=np.array(a2)
    aa1,aa2=np.meshgrid(a1,a2)
    aa1,aa2=aa1[(np.abs(aa2)<1)&(np.abs(aa1)<(1-aa2-precision))],aa2[(np.abs(aa2)<1)&(np.abs(aa1)<(1-aa2-precision))]
    aa1, aa2=aa1*gmpy2.mpfr(1),aa2*gmpy2.mpfr(1)
    Fs2lst=[]
    if q==1:
        pass
    elif q==2:
        pass
    elif q==3:
        for s in S:
            Fs2 = ar2dfa3_MP(np.float64(s),aa1,aa2)
            Fs2lst.append(Fs2)
        Fs2lst = np.array(Fs2lst).swapaxes(0,-1)
    return aa1.astype(np.float64),aa2.astype(np.float64),Fs2lst.astype(np.float64)

def AR2DFA_MP(S,a1=None,a2=None,q=3,precision=0.01):
    '''
    The square of DFA fluctuation function of standard AR(2) process
    based on MP（Multiple Precision math:mpmath）

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a1 : NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    a2 : NoneType,list or array_like,other Iterable eta
    Select special a list for second-order autoregressive coefficient grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    tuple
    (
    aa1 : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process)

    aa2 : numpy.ndarray
    Grid of second-order autoregressive coefficient(AR(2) process)

    Fs2 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process
    )
    '''
    import mpmath as mp
    mp.mp.prec=int(1/precision)
    if a1 is None:
        a1 = np.arange(-2 + precision, 2, precision)
    else:
        a1=np.array(a1)
    if a2 is None:
        a2 = np.arange(-1+precision,1,precision)
    else:
        a2=np.array(a2)
    aa1,aa2=np.meshgrid(a1,a2)
    aa1,aa2=aa1[(np.abs(aa2)<1)&(np.abs(aa1)<(1-aa2-precision))],aa2[(np.abs(aa2)<1)&(np.abs(aa1)<(1-aa2-precision))]
    aa1, aa2=aa1*mp.mpf(1),aa2*mp.mpf(1)
    Fs2lst=[]
    if q==1:
        pass
    elif q==2:
        pass
    elif q==3:
        for s in S:
            Fs2 = ar2dfa3_MP(np.float64(s),aa1,aa2)
            Fs2lst.append(Fs2)
        Fs2lst = np.array(Fs2lst).swapaxes(0,-1)
    return aa1.astype(np.float64),aa2.astype(np.float64),Fs2lst.astype(np.float64)

def AR2DFA_(S,a1=None,a2=None,q=3,precision=0.01):
    '''
    The square of DFA fluctuation function of standard AR(2) process

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a1 : NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    a2 : NoneType,list or array_like,other Iterable eta
    Select special a list for second-order autoregressive coefficient grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    aa1 : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).

    aa2 : numpy.ndarray
    Grid of second-order autoregressive coefficient(AR(2) process).

    Fs2p_AR2 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    '''


    if a1 is None:
        a1 = np.arange(-2 + precision, 2, precision)
    else:
        a1=np.array(a1)
    if a2 is None:
        a2 = np.arange(-1+precision,1,precision)
    else:
        a2=np.array(a2)
    aa1,aa2=np.meshgrid(a1,a2)
    aa1,aa2=aa1[(np.abs(aa2)<1)&(np.abs(aa1)<(1-aa2-precision))],aa2[(np.abs(aa2)<1)&(np.abs(aa1)<(1-aa2-precision))]
    aa1, aa2=aa1.astype(np.float64),aa2.astype(np.float64)
    Fs2lst=[]
    if q==1:
        pass
    elif q==2:
        pass
    elif q==3:
        for s in S:
            Fs2 = ar2dfa3(np.float64(s),aa1,aa2)
            Fs2lst.append(Fs2)
        Fs2lst = np.array(Fs2lst).swapaxes(0,-1)
    return aa1[~np.isnan(Fs2lst.mean(axis=1))],aa2[~np.isnan(Fs2lst.mean(axis=1))],Fs2lst[~np.isnan(Fs2lst.mean(axis=1))]

def AR2DFA(S,a1=None,a2=None,q=3,precision=0.01,GMP=False,MP=False,pre_AR2DFA_dict=None):
    '''
    DFA fluctuation function of standard AR(2) process.

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a1 : NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    a2 : NoneType,list or array_like,other Iterable eta
    Select special a list for second-order autoregressive coefficient grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    pre_AR2DFA_dict : NoneType, dict
    Prepared dictionary.

    Returns
    -------
    tuple
    (
    aa1 : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).
    aa2 : numpy.ndarray
    Grid of second-order autoregressive coefficient(AR(2) process).
    Fs2p_AR2 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    )

    Reference
    -------
    Meyer, P. G., & Kantz, H. (2019). Inferring characteristic timescales from the effect of autoregressive dynamics on detrended fluctuation analysis.
    New Journal of Physics, 21(3), 33022. https://doi.org/10.1088/1367-2630/ab0a8a
    '''
    if pre_AR2DFA_dict is None:
        if GMP == False and MP == False:
            aa1, aa2, Fs2p_AR2 = AR2DFA_(S, a1=a1, a2=a2, q=q, precision=precision)
        elif GMP == True:
            aa1, aa2, Fs2p_AR2 = AR2DFA_GMP(S, a1=a1, a2=a2, q=q, precision=precision)
        elif MP == True:
            aa1, aa2, Fs2p_AR2 = AR2DFA_GMP(S, a1=a1, a2=a2, q=q, precision=precision)
    elif pre_AR2DFA_dict is True:
        tempdict=get_pre_AR2DFA_dict(S)
        aa1, aa2, Fs2p_AR2 = tempdict['aa1'], tempdict['aa2'], tempdict['Fs2p_AR2']
    else:
        aa1, aa2, Fs2p_AR2 = pre_AR2DFA_dict['aa1'], pre_AR2DFA_dict['aa2'], pre_AR2DFA_dict['Fs2p_AR2']
    return aa1, aa2, Fs2p_AR2

def generate_pre_AR2DFA_dict(S,q=3,precision=0.01, GMP=True,MP=False,save_nc=False):
    '''
    Calculate dict: square of DFA fluctuation function of standard AR(2) process

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    save_nc : bool
    Saved dict to netCDF4 format in local dir.

    Returns
    -------
    dict
    {
    aa1 : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process)

    aa2 : numpy.ndarray
    Grid of second-order autoregressive coefficient(AR(2) process)

    Fs2p_AR2 : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process
    }
    '''
    import xarray as xr
    aa1, aa2, Fs2p_AR2 = AR2DFA(S=S,q=q,precision=precision, GMP=GMP,MP=MP)
    if save_nc is True:
        ds = xr.Dataset(
            {
                "aa1": (["ar2index"], aa1),
                "aa2": (["ar2index"], aa2),
                "Fs2p_AR2": (["ar2index", "S"], Fs2p_AR2),
            },
            coords={
                "ar2index": np.arange(len(aa1)),
                "S": S,
            },
            attrs=dict(description="Fs2p_AR2 data.")
        )
        ds['aa1'].attrs = {"units": 1, 'long_name': "first-order autoregressive coefficient for AR(2) grid search"}
        ds['aa2'].attrs = {"units": 1, 'long_name': "second-order autoregressive coefficient for AR(2) grid search"}
        ds['aa2'].attrs = {"units": 1, 'long_name': "The square of DFA fluctuation function of standard AR(2) process"}
        ds.to_netcdf("Fs2p_AR2.nc")
        ds.close()
    else:
        pass
    pre_AR2DFA_dict={"aa1": aa1,
                     "aa2": aa2,
                     "Fs2p_AR2":Fs2p_AR2}
    return pre_AR2DFA_dict


def Lqdfa3(tau,s):
    '''
    Autocorrelation iteration coefficient

    Parameters
    ----------
    tau : numpy.ndarray
    s : numpy.ndarray

    Returns
    -------
    Lq : numpy.ndarray
    Autocorrelation iteration coefficient array.

    '''

    Lq = (4 * s ** 9 - 63 * s ** 8 * tau - 42 * s ** 6 * tau * (-29 + 8 * tau ** 2) + 12 * s ** 7 * (
                -10 + 21 * tau ** 2) - 84 * s ** 5 * (-13 + 42 * tau ** 2) - 144 * s * (
                      -16 + 63 * tau ** 2) + 4 * s ** 3 * (-820 + 3087 * tau ** 2) + 21 * s ** 4 * tau * (
                      -323 + 164 * tau ** 2 + 12 * tau ** 4) - 12 * s ** 2 * tau * (
                      -901 + 595 * tau ** 2 + 105 * tau ** 4 + 12 * tau ** 6) + tau * (
                      -2304 + 1540 * tau ** 2 + 483 * tau ** 4 + 246 * tau ** 6 + 35 * tau ** 8)) / (
                     126 * s ** 2 * (-36 + 49 * s ** 2 - 14 * s ** 4 + s ** 6))
    return Lq

def Lqdfa(q,tau,s,GMP=False,precision=0.01):
    if GMP is True:
        import gmpy2
        gmpy2.get_context().precision = int(1 / precision)
        tau = tau / gmpy2.mpfr(1)
        s = s / gmpy2.mpfr(1)
    else:
        pass
    if q==0:
        Lq = (-tau ** 3
              + 3 * tau ** 2 * s
              + (-3 * s ** 2 + 1) * tau
              + s ** 3 - s) / (6 * s ** 2)
    elif q==1:
        Lq = (3*tau**5
              +(-20*s**2+5)*tau**3
              +30*(s**3-s)*tau**2
              +(-15*s**4+35*s**2-8)*tau
              +2*s**5-10*s**3+8*s)/(30*(s**4-s**2))

    elif q==2:
        Lq = (-10*tau**7
              +(42*s**2-28)*tau**5
              -35*(3*s**4-9*s**2+2)*tau**3
              +105*(s**5-5*s**3+4*s)*tau**2
              +(-35*s**6+280*s**4-497*s**2+108)*tau
              +3*(s**7-14*s**5+49*s**3-36*s)
              )/(70*(s**6-5*s**4+4*s**2))

    elif q==3:
        Lq = (4 * s ** 9 - 63 * s ** 8 * tau - 42 * s ** 6 * tau * (-29 + 8 * tau ** 2) + 12 * s ** 7 * (
                -10 + 21 * tau ** 2) - 84 * s ** 5 * (-13 + 42 * tau ** 2) - 144 * s * (
                      -16 + 63 * tau ** 2) + 4 * s ** 3 * (-820 + 3087 * tau ** 2) + 21 * s ** 4 * tau * (
                      -323 + 164 * tau ** 2 + 12 * tau ** 4) - 12 * s ** 2 * tau * (
                      -901 + 595 * tau ** 2 + 105 * tau ** 4 + 12 * tau ** 6) + tau * (
                      -2304 + 1540 * tau ** 2 + 483 * tau ** 4 + 246 * tau ** 6 + 35 * tau ** 8)) / (
                     126 * s ** 2 * (-36 + 49 * s ** 2 - 14 * s ** 4 + s ** 6))

    return Lq.astype(np.float64)

def ARFIMADFA_GMP(S,a=None,d=None,q=3,precision=0.01):
    '''
    Integrated methods for square of DFA fluctuation function of standard AR(2) process
    based on GMP（GNU Multiple Precision library:gmpy2）

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a :  NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    d : NoneType,list or array_like,other Iterable eta
    Select special a list for dfrac-order grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    tuple
    (
    dd : numpy.ndarray
    Grid of dfrac coefficient(AR(2) process).

    aa : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).

    Fs2p_ARFIMA : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    )
    '''

    import gmpy2
    gmpy2.get_context().precision = int(1 / precision)
    if d is None:
        d = np.arange(-0.5 + precision, 0.5, precision)
    else:
        d = np.array(d)
    if a is None:
        a = np.array([0])
    else:
        a = np.array([0])

    aa = np.expand_dims(a, axis=-1)
    consta = np.ones_like(aa)

    dd = np.expand_dims(d, axis=0)
    constd = np.ones_like(dd)

    aa = np.matmul(aa, constd).astype(np.float64)
    dd = np.matmul(consta, dd).astype(np.float64)
    dd,aa=dd.reshape(-1),aa.reshape(-1)
    if q <= 2:
        Fs2p_ARFIMA = []
        for d in dd:
            Fs2p_ARFIMA_temp = []
            for s in S:
                Fs2p_ARFIMA_temp.append(arfimadfa_GMP(q,s, d))
            Fs2p_ARFIMA.append(Fs2p_ARFIMA_temp)

    elif q==3:
        Fs2p_ARFIMA = []
        for d in dd:
            Fs2p_ARFIMA_temp = []
            for s in S:
                Fs2p_ARFIMA_temp.append(arfimadfa3_GMP(s, d))
            Fs2p_ARFIMA.append(Fs2p_ARFIMA_temp)

    return dd,aa,np.array(Fs2p_ARFIMA).astype(np.float64)

def ARFIMADFA_GMP__(S,a=None,d=None,q=3,precision=0.01):
    '''
    Integrated methods for square of DFA fluctuation function of standard AR(2) process
    based on GMP（GNU Multiple Precision library:gmpy2）

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a :  NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    d : NoneType,list or array_like,other Iterable eta
    Select special a list for dfrac-order grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    tuple
    (
    dd : numpy.ndarray
    Grid of dfrac coefficient(AR(2) process).

    aa : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).

    Fs2p_ARFIMA : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    )
    '''

    import gmpy2
    gmpy2.get_context().precision = int(1 / precision)
    if d is None:
        d = np.arange(-0.5 + precision, 0.5, precision)
    else:
        d = np.array(d)
    if a is None:
        a = np.array([0])
    else:
        pass

    aa = np.expand_dims(a, axis=-1)
    consta = np.ones_like(aa)

    dd = np.expand_dims(d, axis=0)
    constd = np.ones_like(dd)

    aa = np.matmul(aa, constd).astype(np.float64)
    dd = np.matmul(consta, dd).astype(np.float64)
    dd,aa=dd.reshape(-1),aa.reshape(-1)
    if q <= 2:
        Fs2p_ARFIMA = []
        for d in dd:
            Fs2p_ARFIMA_temp = []
            for s in S:
                Fs2p_ARFIMA_temp.append(arfimadfa_GMP(q,s, d))
            Fs2p_ARFIMA.append(Fs2p_ARFIMA_temp)

    elif q==3:
        Fs2p_ARFIMA = []
        for d in dd:
            Fs2p_ARFIMA_temp = []
            for s in S:
                Fs2p_ARFIMA_temp.append(arfimadfa3_GMP_(aa,s, d))
            Fs2p_ARFIMA.append(Fs2p_ARFIMA_temp)

    return dd,aa,np.array(Fs2p_ARFIMA).astype(np.float64)

def ARFIMADFA_MP(S,a=None,d=None,q=3,precision=0.01):
    '''
    Integrated methods for square of DFA fluctuation function of standard AR(2) process
    based on MP（Multiple Precision Math:mpmath）

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a :  NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    d : NoneType,list or array_like,other Iterable eta
    Select special a list for dfrac-order grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    tuple
    (
    dd : numpy.ndarray
    Grid of dfrac coefficient(AR(2) process).

    aa : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).

    Fs2p_ARFIMA : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    )
    '''
    import mpmath as mp
    mp.mp.prec = int(1 / precision)
    if d is None:
        d = np.arange(-0.5 + precision, 0.5, precision)
    else:
        d = np.array(d)
    if a is None:
        a = np.array([0])
    else:
        a = np.array(a)

    aa = np.expand_dims(a, axis=-1)
    consta = np.ones_like(aa)

    dd = np.expand_dims(d, axis=0)
    constd = np.ones_like(dd)

    aa = np.matmul(aa, constd).astype(np.float64)
    dd = np.matmul(consta, dd).astype(np.float64)
    ddi = (dd + 1j * aa).astype(np.complex128)
    dd, aa ,ddi = dd.reshape(-1), aa.reshape(-1),ddi.reshape(-1)

    if q <= 2:
        Fs2p_ARFIMA = []
        for di in ddi:
            Fs2p_ARFIMA_temp = []
            for s in S:
                Fs2p_ARFIMA_temp.append(arfimadfa_MP(q,s, di))
            Fs2p_ARFIMA.append(Fs2p_ARFIMA_temp)

    elif q == 3:
        Fs2p_ARFIMA = []
        for di in ddi:
            Fs2p_ARFIMA_temp = []
            for s in S:
                Fs2p_ARFIMA_temp.append(arfimadfa3_MP(s, di))
            Fs2p_ARFIMA.append(Fs2p_ARFIMA_temp)

    return dd, aa, np.array(Fs2p_ARFIMA).astype(np.float64)

def ARFIMADFA_(S,a=None,d=None,q=3,precision=0.01):
    '''
    Integrated methods for square of DFA fluctuation function of standard AR(2) process.

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a :  NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    d : NoneType,list or array_like,other Iterable eta
    Select special a list for dfrac-order grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    Returns
    -------
    tuple
    (
    dd : numpy.ndarray
    Grid of dfrac coefficient(AR(2) process).

    aa : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).

    Fs2p_ARFIMA : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    )
    '''
    from scipy.special import gamma
    if d is None:
        d = np.arange(-0.5 + precision, 0.5, precision)
    else:
        d = np.array(d)
    if a is None:
        a = np.array([0])
    else:
        a = np.array(a)
    aa = np.expand_dims(a, axis=-1)
    consta = np.ones_like(aa)

    dd = np.expand_dims(d, axis=0)
    constd = np.ones_like(dd)

    aa = np.matmul(aa, constd).astype(np.float64)
    dd = np.matmul(consta, dd).astype(np.float64)
    ddi=(dd+1j*aa).astype(np.complex128)

    Fs2lst=[]

    if q<=2:
        for s in S:
            Fs2 = Lqdfa(q=q, tau=0,s=float(s))
            for t in range(1, s):
                C = (gamma(1 - ddi)) / (gamma(ddi)) * t ** (2 * d - 1)
                Fs2 = Fs2 + 2 * C * Lqdfa(q=q, tau=t, s=float(s))

            Fs2lst.append(np.abs(Fs2))

    elif q==3:
        for s in S:

            Fs2 = Lqdfa3(0, float(s))
            for t in range(1, s):
                C = (gamma(1 - ddi)) / (gamma(ddi)) * t ** (2 * d - 1)
                Fs2 = Fs2 + 2 * C * Lqdfa3(t, float(s))

            Fs2lst.append(np.abs(Fs2))

    Fs2lst = (np.array(Fs2lst).swapaxes(0,-1))

    return dd.reshape(-1),aa.reshape(-1),Fs2lst.reshape(-1,len(S))

def arfimadfa3_GMP(s, d):
    import gmpy2

    ss=s
    s = s / gmpy2.mpfr(1)
    d = d / gmpy2.mpfr(1)
    Fs2 = Lqdfa3(0, s)
    for t in range(1, ss):
        C = (gmpy2.gamma(1 - d)) / (gmpy2.gamma(d)) * t ** (2 * d - 1)
        Fs2 = Fs2 + 2 * C * Lqdfa3(t, s)
    return Fs2


def arfimadfa3_GMP_(a,s, d):
    import gmpy2

    ss=s

    s = s / gmpy2.mpfr(1)
    d = d / gmpy2.mpfr(1)

    Fs2 = Lqdfa3(0, s)

    for t in range(1, ss):
        d1=d-0.5
        d2=-d-1
        C = ((gmpy2.gamma(1 - d))/(gmpy2.gamma(d)*(1+a)**2)
             +(gmpy2.gamma(1 - d1))/gmpy2.gamma(d1)*(1+a)*2*a
             +(gmpy2.gamma(1 - d2))/(gmpy2.gamma(d2)*(a)**2))/(4*a**2+5*a+1) * t ** (2 * d - 1)
        Fs2 = Fs2 + 2 * C * Lqdfa3(t, s)

    return Fs2

def arfimadfa_GMP(q,s, d):
    import gmpy2

    ss=s
    s = s / gmpy2.mpfr(1)
    d = d / gmpy2.mpfr(1)
    Fs2 = Lqdfa(q=q,tau=0, s=s)
    for t in range(1, ss):
        C = (gmpy2.gamma(1 - d)) / (gmpy2.gamma(d)) * t ** (2 * d - 1)
        Fs2 = Fs2 + 2 * C * Lqdfa(q=q,tau=t, s=s)
    return Fs2

def arfimadfa3_MP(s, di):
    import mpmath as mp

    ss = s
    s = s *mp.mpf(1)
    d = di *mp.mpf(1)
    Fs2 = Lqdfa3(0, s)

    for t in range(1, ss):
        C = (mp.gamma(1 - d)) / (mp.gamma(d)) * t ** (2 * d - 1)
        Fs2 = Fs2 + 2 * C * Lqdfa3(t, s)
    return abs(Fs2)

def arfimadfa_MP(q,s, di):
    import mpmath as mp

    ss = s
    s = s *mp.mpf(1)
    d = di *mp.mpf(1)
    Fs2 = Lqdfa(q=q,tau=0, s=s)

    for t in range(1, ss):
        C = (mp.gamma(1 - d)) / (mp.gamma(d)) * t ** (2 * d - 1)
        Fs2 = Fs2 + 2 * C * Lqdfa(q=q,tau=t, s=s)
    return abs(Fs2)

def ARFIMADFA(S,a=None,d=None,q=3,precision=0.01,GMP=False,MP=False,pre_ARFIMADFA_dict=None):
    '''
   square of DFA fluctuation function of standard ARFIMA process.

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    a :  NoneType,list or array_like,other Iterable eta
    Select special a list for first-order autoregressive coefficient grid search.

    d : NoneType,list or array_like,other Iterable eta
    Select special a list for dfrac-order grid search.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    pre_ARFIMADFA_dict : NoneType, dict
    Prepared dictionary.

    Returns
    -------
    tuple
    (
    dd : numpy.ndarray
    Grid of dfrac coefficient(AR(2) process).

    aa : numpy.ndarray
    Grid of first-order autoregressive coefficient(AR(2) process).

    Fs2p_ARFIMA : numpy.ndarray
    The square of DFA fluctuation function of standard AR(2) process.
    )
    '''
    if pre_ARFIMADFA_dict is None:
        if GMP == False and MP == False:
            dd, aa, Fs2p_ARFIMA = ARFIMADFA_(S=S, a=a, d=d, q=q, precision=precision)
        elif a != None and GMP == True:
            dd, aa, Fs2p_ARFIMA = ARFIMADFA_MP(S=S, a=a, d=d, q=q, precision=precision)
            import warnings
            warnings.warn(
                "The scope of the gmpy2's gamma function is not complex,it trun to mpmath automatically ",
                UserWarning)
        elif MP == True:
            dd, aa, Fs2p_ARFIMA = ARFIMADFA_MP(S=S, a=a, d=d, q=q, precision=precision)
        elif a == None and GMP == True:
            dd, aa, Fs2p_ARFIMA = ARFIMADFA_GMP(S=S, a=a, d=d, q=q, precision=precision)
    elif pre_ARFIMADFA_dict is True:
        tempdict=get_pre_ARFIMADFA_dict(S)
        dd, aa, Fs2p_ARFIMA = tempdict['dd'], tempdict['aa'], tempdict['Fs2p_ARFIMA']
    else:
        dd, aa, Fs2p_ARFIMA =pre_ARFIMADFA_dict['dd'],pre_ARFIMADFA_dict['aa'],pre_ARFIMADFA_dict['Fs2p_ARFIMA']

    return dd,aa,Fs2p_ARFIMA

def generate_pre_ARFIMADFA_dict(S,q=3,precision=0.01, GMP=True,MP=False,save_nc=False):
    '''
    Calculate dict: square of DFA fluctuation function of standard ARFIMA process

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    q : int
    Order of the detrending polynomial time series or its array.

    precision : float
    Precision of grid search.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    save_nc : bool
    Saved dict to netCDF4 format in local dir.

    Returns
    -------
    dict
    {
    dd : numpy.ndarray
    Grid of dfrac-order coefficient(ARFIMA process)

    aa : numpy.ndarray
    Grid of first-order autoregressive coefficient(ARFIMA process)

    Fs2p_ARFIMA : numpy.ndarray
    The square of DFA fluctuation function of standard ARFIMA process
    }
    '''
    import xarray as xr
    dd, aa, Fs2p_ARFIMA = ARFIMADFA(S=S,q=q,precision=precision, GMP=GMP,MP=MP)
    if save_nc is True:
        ds = xr.Dataset(
            {
                "dd": (["arfimaindex"], dd),
                "aa": (["arfimaindex"], aa),
                "Fs2p_ARFIMA": (["arfimaindex", "S"], Fs2p_ARFIMA),
            },
            coords={
                "arfimaindex": np.arange(len(dd)),
                "S": S,
            },
            attrs=dict(description="Fs2p_ARFIMA data.")
        )
        ds['dd'].attrs = {"units": 1, 'long_name': "dfrac-order coefficient for ARFIMA grid search"}
        ds['aa'].attrs = {"units": 1, 'long_name': "first-order autoregressive coefficient for ARFIMA grid search"}
        ds.to_netcdf("Fs2p_ARFIMA(0,d,0).nc")
        ds.close()
    else:
        pass
    pre_ARFIMADFA_dict={"dd": dd,
                     "aa": aa,
                     "Fs2p_ARFIMA":Fs2p_ARFIMA}
    return pre_ARFIMADFA_dict

def get_pre_ARFIMADFA_dict(S,path='Fs2p_ARFIMA(0,d,0).nc'):
    '''
    Get prepared ARFIMADFA(dict) from local netCDF4 file.

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    path : str
    Filename and its dir.

    Returns
    -------
    dict

    '''
    import xarray as xr
    ds = xr.open_dataset(path, decode_cf=False).sel(S=S)
    pre_AR1DFA_dict = {'dd': np.array(ds['dd'][:]),
                       'aa': np.array(ds['aa'][:]),
                       'Fs2p_ARFIMA': np.array(ds['Fs2p_ARFIMA'][:])}
    ds.close()
    return pre_AR1DFA_dict

def get_pre_AR2DFA_dict(S,path='Fs2p_AR2.nc'):
    '''
    Get prepared AR2DFA(dict) from local netCDF4 file.

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    path : str
    Filename and its dir.

    Returns
    -------
    dict

    '''
    import xarray as xr
    ds = xr.open_dataset(path, decode_cf=False).sel(S=S)
    pre_AR2DFA_dict = {'aa1': np.array(ds['aa1'][:]), 'aa2': np.array(ds['aa2'][:]),
                       'Fs2p_AR2': np.array(ds['Fs2p_AR2'][:])}
    ds.close()
    return pre_AR2DFA_dict

def get_pre_AR1DFA_dict(S,path='Fs2p_AR1.nc'):
    '''
    Get prepared AR1DFA(dict) from local netCDF4 file.

    Parameters
    ----------
    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    path : str
    Filename and its dir.

    Returns
    -------
    dict

    '''
    import xarray as xr
    ds = xr.open_dataset(path, decode_cf=False).sel(S=S)
    pre_AR1DFA_dict = {'a': np.array(ds['a'][:]),
                       'Fs2p_AR1': np.array(ds['Fs2p_AR1'][:])}
    ds.close()
    return pre_AR1DFA_dict