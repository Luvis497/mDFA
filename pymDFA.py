# -*- coding: utf-8 -*-
# -------------------------------

# @product：PyCharm

# -------------------------------

# @filename：pymDFA.py
# @teim：2025/11/1 9:59
# @email：2301110293@pku.edu.cn

# -------------------------------


import numpy as np
import dask
import dask.array as da
import xarray as xr

def My_polydel(x,y,q=1,axis=-1,choice=None):

    '''
    My_polydel5 is to obtain residual after polynomial fitting(V4).

    Parameters
    ----------
    x: numpy.ndarray,list or other Iterable
    The independent variable of polynomial fitting.
    y: numpy.ndarray,list or other Iterable
    The dependent variable of polynomial fitting.
    q: int
    Order of polynomial fitting.
    axis: int
    Axis of function action.
    choice: numpy.ndarray,list or other Iterable
    Select special order of selected polynomial fitting.

    Returns
    -------
    Res : numpy.ndarray
    residual after polynomial fitting
    '''

    if choice is not None:
        qlst = choice
    else:
        qlst = np.arange(q + 1)

    x = x.swapaxes(-1, axis)
    y = y.swapaxes(-1, axis)
    x.astype(np.float64)
    y.astype(np.float64)
    X = np.stack([x ** _ for _ in qlst], axis=-1)
    Y = np.expand_dims(y, axis=-1)

    Res = Y- np.matmul(X,np.matmul(np.linalg.pinv(X),Y))

    return Res.squeeze(axis=-1)

def DFA(X,q=3,S=None,axis=0,type="double",filter=True,**kwargs):  #detrended fluctuation analysis
    '''
    DFA(detrended fluctuation analysis) method
    used to confirm power-law and autocorrelation relationships.

    Parameters
    ----------
    X : numpy.ndarray,list or other Iterable
    Time series or its array.
    q : int
    Order of the detrending polynomial time series or its array.
    S : NoneType list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA
    axis : int, optional
    By default, the index is into the flattened array, otherwise
    along the specified axis.
    type : string
    type of segmentation. "right":Ignore data from the residual window on the left,
    "left":Ignore data from the residual window on the right,
    "double":use both sides of data,but it requires two iterations,
    "0.95th":residual windows less than 0.05th of data,but it requires more iterations,
    "0.99th":residual windows less than 0.01th of data,but it requires more iterations.
    filter: bool
    The DFA method is so complex that F(s) unreliable  when s of F(s) is small(usually s<30) and absense of Multiple Precision.

    Returns
    -------
    S : numpy.ndarray
    Sequence of segment numbers, variable of DFA
    Fs2 : numpy.ndarray
    The square of DFA fluctuation function

    Reference
    -------
    Höll, M., & Kantz, H. (2015). The relationship between the detrendend fluctuation analysis and the autocorrelation function of a signal.
    The European Physical Journal. B, Condensed Matter Physics, 88(12), 1-7. https://doi.org/10.1140/epjb/e2015-60721-1
    '''

    X = np.array(X).astype(np.float64)
    X=X.swapaxes(axis,0)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    Y=np.cumsum(X,axis=0)
    L=len(Y)

    if S is None:
        S = np.hstack((np.arange(8, 12), np.unique((10 ** np.arange(1.1, np.log10(L / 4), 0.01)).astype(np.int64))))
    else:
        S=np.array(S)

    if filter is True:
        filter_mode = kwargs.pop('filter_mode', None)
        if filter_mode == None:
            S = S[S > 25]
        else:
            s1, s2 = filter_mode[:]
            if s1 <= 1:
                s1 = int(L * s1)
            else:
                pass
            if s2 <= 1:
                s2 = int(L * s2)
            S = S[(S > s1) & (S < s2)]

    #S=S[S>=100]
    #S=np.arange(2*q + 3,min(L//80,100))
    #S = np.arange(9,  100)
    Fs2=[]

    for s in S:
        r = L % s
        if type=="left":
            matrixs = Y[:L-r].reshape(L//s,s,*Y.shape[1:])
        elif type=="right":
            matrixs = Y[r:].reshape(L//s,s,*Y.shape[1:])
        elif type=="double":
            matrixs = np.stack((Y[:len(Y)-r].reshape(L//s,s,*Y.shape[1:]), Y[r:].reshape(L//s,s,*Y.shape[1:])),axis=0).reshape(-1,s,*Y.shape[1:])
        else:
            pass

        y=matrixs.swapaxes(1,-1).astype(np.float64)
        #y = np.matmul(matrixs, np.triu(np.ones((s,s)),0))
        v = np.tile(np.arange(s), list(y.shape)[:-1] + [1]).astype(np.float64)
        z=My_polydel(v,y,q)
        z=z.swapaxes(1,-1)
        Fs2.append((z**2).mean(axis=(0,1)))
        #print(len(Fs2)/len(S))
    return S,np.array(Fs2)

def DFA_da(X, q=3, S=None, axis=0, type="double", filter=True, **kwargs):  # detrended fluctuation analysis
    '''
    DFA(detrended fluctuation analysis) method
    used to confirm power-law and autocorrelation relationships.
    based on Dynamic programming calculation(dask).

    Parameters
    ----------
    X : numpy.ndarray,list or other Iterable
    Time series or its array.
    q : int
    Order of the detrending polynomial time series or its array.
    S : NoneType,list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA
    axis : int, optional
    By default, the index is into the flattened array, otherwise
    along the specified axis.
    type : string
    type of segmentation. "right":Ignore data from the residual window on the left,
    "left":Ignore data from the residual window on the right,
    "double":use both sides of data,but it requires two iterations,
    "0.95th":residual windows less than 0.05th of data,but it requires more iterations,
    "0.99th":residual windows less than 0.01th of data,but it requires more iterations.
    filter: bool
    The DFA method is so complex that F(s) unreliable  when s of F(s) is small(usually s<30) and absense of Multiple Precision.

    Returns
    -------
    S : numpy.ndarray
    Sequence of segment numbers, variable of DFA
    Fs2 : numpy.ndarray
    The square of DFA fluctuation function

    Reference
    -------
    Höll, M., & Kantz, H. (2015). The relationship between the detrendend fluctuation analysis and the autocorrelation function of a signal.
    The European Physical Journal. B, Condensed Matter Physics, 88(12), 1-7. https://doi.org/10.1140/epjb/e2015-60721-1

    '''
    standard_=kwargs.pop('standard',True)
    diff_order = kwargs.pop('diff_order', 0)
    if standard_ is True:
        X = np.array(X).astype(np.float64)
        X = X.swapaxes(axis, 0)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    else:
        pass

    L = len(X)

    if S is None:
        S_max = kwargs.pop('S_max', False)
        if S_max is True:
            S = np.hstack(
                (np.arange(8, 12), np.unique((10 ** np.arange(1.1, np.log10(L / 4), 0.01)).astype(np.int64))))
        else:
            S = np.hstack(
                (np.arange(8, 12), np.unique((10 ** np.arange(1.1, np.log10(L / 10), 0.01)).astype(np.int64))))

        if filter is True:
            filter_mode = kwargs.pop('filter_mode', None)
            if filter_mode == None:
                S = S[S > 25]

            elif filter_mode == 'sparse':
                S = np.unique((10 ** np.arange(1.1, np.log10(L / 10), 0.01)).astype(np.int64))

            else:
                s1, s2 = filter_mode[:]
                if s1 <= 1:
                    s1 = int(L * s1)
                else:
                    pass
                if s2 <= 1:
                    s2 = int(L * s2)
                S = S[(S > s1) & (S < s2)]

        else:
            pass
    else:
        S = np.array(S).astype(np.int64)

    divide = kwargs.pop('divide', False)
    block_size=kwargs.pop('block_size',5000000)
    if (divide is True) or (X.size > block_size):
        import warnings
        warnings.warn(
            "Input data is too large and must be divided to calculation",
            RuntimeWarning)

        shape0=X.shape

        X=X.reshape(shape0[0],-1)
        Len_X=X.shape[-1]

        if Len_X<2:
            pass
        else:
            split_X=Len_X//2
            Fs2=np.hstack([DFA_da(X=X[:,:split_X],q=q,S=S,type=type,standard=False,diff_order=diff_order)[1],DFA_da(X=X[:,split_X:],q=q,S=S,type=type,standard=False,diff_order=diff_order)[1]])
            Fs2=Fs2.reshape(-1,*list(shape0)[1:])

            return S,Fs2
    else:
        pass

    if diff_order == 0:
        Y = np.cumsum(X, axis=0)
    elif diff_order == 1:
        X_std = (X[1:] - X[:-1]).std(axis=0)
        #print('diff_order=1')
        Y = X / X_std
    else:
        Y = X
        for _ in range(diff_order - 1):
            Y = Y[1:] - Y[:-1]

    Fs2 = []

    for s in S:
        r = L % s
        m = L // s

        if type == "left":
            matrixs = Y[:L - r].reshape(m, s, *Y.shape[1:])
        elif type == "right":
            matrixs = Y[r:].reshape(m, s, *Y.shape[1:])
        elif type == "double":
            matrixs = np.stack(
                (Y[:len(Y) - r].reshape(m, s, *Y.shape[1:]), Y[r:].reshape(m, s, *Y.shape[1:])),
                axis=0).reshape(-1, s, *Y.shape[1:])
        elif type == "0.95th":
            if r < (L / 10):
                matrixs = np.stack(
                    (Y[:len(Y) - r].reshape(m, s, *Y.shape[1:]), Y[r:].reshape(m, s, *Y.shape[1:])),
                    axis=0).reshape(-1, s, *Y.shape[1:])
            else:
                rr = (r // int(L / 20)) // 2 + 1
                mm = int(L / 20)
                matrixs = np.stack(
                    [Y[i * mm:len(Y) - r + i * mm].reshape(m, s, *Y.shape[1:]) for i in range(rr)] + [
                        Y[r - i * mm:len(Y) - i * mm].reshape(m, s, *Y.shape[1:]) for i in range(rr)],
                    axis=0).reshape(-1, s, *Y.shape[1:])
        elif type == "0.99th":
            if r < (L / 50):
                matrixs = np.stack(
                    (Y[:len(Y) - r].reshape(m, s, *Y.shape[1:]), Y[r:].reshape(m, s, *Y.shape[1:])),
                    axis=0).reshape(-1, s, *Y.shape[1:])
            else:
                rr = (r // int(L / 100)) // 2 + 1
                mm = int(L / 100)
                matrixs = np.stack(
                    [Y[i * mm:len(Y) - r + i * mm].reshape(m, s, *Y.shape[1:]) for i in range(rr)] + [
                        Y[r - i * mm:len(Y) - i * mm].reshape(m, s, *Y.shape[1:]) for i in range(rr)],
                    axis=0).reshape(-1, s, *Y.shape[1:])
        elif type == "0.995th":
            if r < (L / 100):
                matrixs = np.stack(
                    (Y[:len(Y) - r].reshape(m, s, *Y.shape[1:]), Y[r:].reshape(m, s, *Y.shape[1:])),
                    axis=0).reshape(-1, s, *Y.shape[1:])
            else:
                rr = (r // int(L / 200)) // 2 + 1
                mm = int(L / 200)
                matrixs = np.stack(
                    [Y[i * mm:len(Y) - r + i * mm].reshape(m, s, *Y.shape[1:]) for i in range(rr)] + [
                        Y[r - i * mm:len(Y) - i * mm].reshape(m, s, *Y.shape[1:]) for i in range(rr)],
                    axis=0).reshape(-1, s, *Y.shape[1:])
        elif type == "all_slide":
            matrixs = np.stack(
                [Y[i:s + i] for i in range(L - s)],
                axis=0).reshape(-1, s, *Y.shape[1:])

        y = matrixs.swapaxes(1, -1).astype(np.float64)
        # y = np.matmul(matrixs, np.triu(np.ones((s,s)),0))
        v = np.tile(np.arange(s), list(y.shape)[:-1] + [1]).astype(np.float64)
        z = dask.delayed(My_polydel)(v, y, q)
        z = (z.swapaxes(1, -1) ** 2).mean(axis=(0, 1))
        Fs2.append(z)
        # print(len(Fs2)/len(S))

    return S, np.array(dask.compute(Fs2)).squeeze(axis=0)

def loss_function(result,target,loss='SR',axis=0,**kwargs):
    if loss=='SR': #similarity ratio
        power=kwargs.get("power",1)
        return ((result / target) ** power).std(axis=axis)
    elif loss=='log(SR)':
        power = kwargs.get("power", 1)
        return ((np.log(result / target) ** power).std(axis=axis))
    elif loss=='SRlog(SR)':
        power = kwargs.get("power", 1)
        return ((((result / target)*np.log(result / target)) ** power).std(axis=axis))

def loss_function_da(result, target, loss='SR', axis=0, **kwargs):
    if loss == 'SR':  # similarity ratio
        power = kwargs.get("power", 2)
        weight = kwargs.get("weight", 1)
        regularization = kwargs.get("regularization", False)
        temp = result / target
        temp = temp - temp.mean(axis=axis, keepdims=True)
        if (regularization is True) :
            re_param = kwargs.get("re_param", 0.2)
            temp2=da.roll(temp,shift=1,axis=axis)*temp
            r= da.where(temp2<=0,1,0).mean(axis=axis)*re_param
        else :
            r= 0
        return ((temp * weight) ** power).mean(axis=axis)*(1-r)
    elif loss == 'log(SR)':
        power = kwargs.get("power", 6)
        weight = kwargs.get("weight", 1)
        regularization = kwargs.get("regularization", False)
        temp = da.log(result / target)
        temp = temp - temp.mean(axis=axis, keepdims=True)
        if (regularization is True) :
            re_param = kwargs.get("re_param", 0.2)
            temp2=da.roll(temp,shift=1,axis=axis)*temp
            r= da.where(temp2<=0,1,0).mean(axis=axis)*re_param
        else :
            r= 0
        return (da.abs(temp * weight) ** power).mean(axis=axis)
    elif loss == 'log(SR)/SR':
        power = kwargs.get("power", 6)
        weight = kwargs.get("weight", 1)
        regularization = kwargs.get("regularization", False)
        temp = da.log(result / target)/(result / target)
        temp = temp - temp.mean(axis=axis, keepdims=True)
        if (regularization is True):
            re_param = kwargs.get("re_param", 0.2)
            temp2 = da.roll(temp, shift=1, axis=axis) * temp
            r = da.where(temp2 <= 0, 1, 0).mean(axis=axis) * re_param
        else:
            r = 0
        return (da.abs(temp * weight) ** power).mean(axis=axis)
    elif loss == 'SRlog(SR)':
        power = kwargs.get("power", 2)
        weight = kwargs.get("weight", 1)
        temp = result / target
        temp = temp * da.log(temp)
        temp = temp - temp.mean(axis=axis, keepdims=True)
        return (((temp * weight) ** power).std(axis=axis))

def get_pre_dict(S, path, grid_values, value='Fs2p', m=1,interp='linear'):
    """
    读取并插值加密ARFIMA拟合数据文件，返回所有变量展平成一维，便于DFA未知过程拟合。

    Parameters
    ----------
    S : list or array_like
        选定的S轴（最后一维）序列。
    path : str
        netCDF文件路径。
    grid_values : list/tuple/str
        需要返回的其它网格变量名（如['a_value', 'd_value']）。
    value : str
        目标变量名。
    m : int
        网格加密倍数，默认1表示不加密。

    Returns
    -------
    pre_dict : dict
        所有变量展平后一一对应，每个长度为“除S轴以外所有轴加密后点数”。
        还有'Fs2p': shape为(N_grid, len(S))，其余变量shape为(N_grid,)。
    """

    ds = xr.open_dataset(path, decode_cf=False).sel(S=S)

    # 找到目标变量的维度和各轴信息
    value_dims = ds[value].dims              # e.g. ('a', 'd', 'S')
    all_axes = list(value_dims)
    s_axis = 'S'                             # 这里假设S轴名为'S'

    # 识别出非S轴
    non_s_axes = [ax for ax in all_axes if ax != s_axis]

    # 准备非S轴新的加密坐标以及grid_shape
    interp_coords = {}                       # {axis_name: new_coords}
    grid_shape = []
    for ax in non_s_axes:
        old_coords = ds[ax].values
        if m > 1:
            num = (len(old_coords) - 1) * m + 1
            new_coords = np.linspace(old_coords[0], old_coords[-1], num)
            interp_coords[ax] = new_coords
            grid_shape.append(num)
        else:
            interp_coords[ax] = old_coords
            grid_shape.append(len(old_coords))
    N_grid = np.prod(grid_shape) if grid_shape else 1

    # 插值目标变量
    da_interp = ds[value]
    if m > 1:
        da_interp = da_interp.interp(interp_coords, method=interp)
    Fs2p = da_interp.values          # shape: (*grid_shape, len(S))

    # reshape
    if Fs2p.ndim > 1:
        Fs2p = Fs2p.reshape(-1, len(S))   # (N_grid, len(S))
    else:
        Fs2p = Fs2p.reshape(-1, )         # (N_grid,)

    pre_dict = {value: Fs2p, 'shape': tuple(grid_shape)}

    # 保证grid_values为list
    if isinstance(grid_values, str):
        grid_values = [grid_values]

    # 生成每个坐标的broadcasted变量（如a_value, d_value等），全部展平
    for idx, ax in enumerate(non_s_axes):
        arr = interp_coords[ax]
        # 关键：reshape为 (1,...,Li,...,1)，Li那轴为原轴长度，其他为1，方便broadcast
        shape = [1] * len(grid_shape)
        shape[idx] = grid_shape[idx]
        arr = arr.reshape(shape)
        arr = np.broadcast_to(arr, grid_shape).reshape(-1)
        # 只有在变量名需要才加入 pre_dict
        if ax in grid_values:
            pre_dict[ax] = arr

    # 其它变量也加进来（如 grid_values 里还有S或其它辅助变量）
    for var in grid_values:
        if var in non_s_axes:          # 已经处理过
            continue
        if var == s_axis:
            # S轴直接给出选用的S
            pre_dict[var] = ds[s_axis].sel(S=S).values
        elif var in ds.variables:
            # 其他辅助指标，注意shape
            arr = ds[var].values
            if arr.shape == tuple(grid_shape):
                arr = arr.reshape(-1)
            pre_dict[var] = arr

    ds.close()
    return pre_dict

def mDFA_fit_process_da(S,Fs2,pre_dict_path,values,axis=0, loss='SR',precision=0.01,GMP=False,MP=False, calculate_residuals=False,
               standard_Fs2=False,return_Fs2=True,kth=0,**kwargs):  # Autoregressive coefficient
    '''
    Integrated methods use DFA(detrended fluctuation analysis) method to fit AR(1) process.
    based on dask.

    Parameters
    ----------
    X : numpy.ndarray,list or other Iterable
    Time series or its array.

    q : int
    Order of the detrending polynomial time series or its array.

    S : list or array_like,other Iterable eta
    Sequence of segment numbers, variable of DFA.

    axis : int, optional
    By default, the index is into the flattened array, otherwise
    along the specified axis.

    type : string
    type of segmentation. "right":Ignore data from the residual window on the left,
    "left":Ignore data from the residual window on the right,
    "double":use both sides of data,but it requires two iterations,
    "0.95th":residual windows less than 0.05th of data,but it requires more iterations,
    "0.99th":residual windows less than 0.01th of data,but it requires more iterations.

    loss : str
    Loss function type.

    precision : float
    Precision of grid search.

    filter: bool
    The DFA method is so complex that F(s) unreliable  when s of F(s) is small(usually s<30) and absense of Multiple Precision.

    GMP : bool
    GMP（GNU Multiple Precision Arithmetic Library:gmpy2） is need or not.

    MP : bool
    MP（Multiple Precision Math:mpmath） is need or not.

    pre_AR1DFA_dict : Nonetype, dict
    Prepared dictionary.

    calculate_residuals : bool
    Calculate the bias or not.

    standard_Fs2 : bool
    Return standarded Fs2 of AR(1) process or not.

    return_Fs2: bool
    Return standarded Fs2 of AR(1) process or not.

    Returns
    -------

    dict

    {
    a_values : numpy.ndarray
    Values of first-order autoregressive coefficient(AR(1) process) fit.

    error_sum_AR1 :  numpy.ndarray or NoneType
    The bias of DFA fit.

    Fs2_AR1_m : numpy.ndarray or NoneType
    standarded Fs2 of AR(1) process.

    Fs2: numpy.ndarray or NoneType
    The square of DFA fluctuation function of real series

    S : numpy.ndarray or NoneType
    Sequence of segment numbers, variable of DFA
    }

    '''
    m=kwargs.pop('m',1)
    interp=kwargs.pop('interp','linear')
    tempdict = get_pre_dict(S=S,path=pre_dict_path,grid_values=values,m=m,interp=interp)
    Fs2p =tempdict['Fs2p']

    Fs2 = Fs2.swapaxes(axis, 0)
    Fs2p = da.from_array(Fs2p)

    Fs2p_m = np.tile(Fs2p, list(Fs2.shape)[1:] + [1, 1])
    order = list(range(len(Fs2p_m.shape)))
    Fs2p_m = Fs2p_m.transpose(order[-2:] + order[:-2])
    if kth==0:

        index = da.argmin((loss_function_da(da.from_array(Fs2), Fs2p_m, loss=loss, axis=1, **kwargs)), axis=0)

        if calculate_residuals:
            error_sum = (loss_function_da(Fs2p[index].swapaxes(0, -1), da.from_array(Fs2), loss=loss, axis=0,
                                          **kwargs)).compute()
        else:
            error_sum = None
        if standard_Fs2:
            Fs2_m = (Fs2p[index].swapaxes(0, -1)).compute()
        else:
            Fs2_m = None
        if return_Fs2:
            pass
        else:
            Fs2 = None
        # index = np.argmin((((Fs2 /Fs2p -1) ** 2).sum(axis=1)), axis=0)
        # index=np.argmin(((np.log(Fs2p/Fs2)**2).sum(axis=1)),axis=0)
        new_dict = {'error_sum': error_sum, 'Fs2_m': Fs2_m, 'Fs2': Fs2, 'S': S}
        index_ = index.compute()
        if isinstance(values, str):
            new_dict[values] = tempdict[values][index_]
        else:
            for v in values:
                new_dict[v] = tempdict[v][index_]



    else:
        #topk-soft-argmax
        l=(loss_function_da(da.from_array(Fs2), Fs2p_m, loss=loss, axis=1, **kwargs))
        indexs = da.argtopk(l,k=-kth, axis=0)

        T = kwargs.pop('T', 2*(da.min(l, axis=0)))
        rT= kwargs.pop('rT',1)
        T=rT*T
        try:
            print(T.compute())
        except Exception as e:
            print(T)


        weights = da.exp(-1*(da.topk(l,k=-kth, axis=0)-da.min(l,axis=0))/T)
        if calculate_residuals:
            error_sum = (weights/weights.sum(axis=0)*(-da.log(weights))).sum(axis=0)*T
        else:
            error_sum = None

        weights = weights/weights.sum(axis=0)


        if standard_Fs2:
            ## ?
            Fs2_m = 0
            for k in range(kth):
                Fs2_m += Fs2p[indexs[k]].swapaxes(0, -1)*(weights[k]).swapaxes(0, -1)
            Fs2_m = Fs2_m.compute()
        else:
            Fs2_m = None
        if return_Fs2:
            pass
        else:
            Fs2 = None
        # index = np.argmin((((Fs2 /Fs2p -1) ** 2).sum(axis=1)), axis=0)
        # index=np.argmin(((np.log(Fs2p/Fs2)**2).sum(axis=1)),axis=0)
        new_dict = {'error_sum': error_sum.compute(), 'Fs2_m': Fs2_m, 'Fs2': Fs2, 'S': S}
        if isinstance(values, str):
            new_dict[values] = (tempdict[values][indexs]*weights).sum(axis=0).compute()
        else:
            for v in values:
                new_dict[v] = (tempdict[v][indexs]*weights).sum(axis=0).compute()
        return_l = kwargs.pop('return_l', False)
        if return_l:
            new_dict['shape']=tempdict['shape']
            new_dict['l'] = l.compute()

    return new_dict












