# -*- coding: utf-8 -*-
# -------------------------------

    # @product：PyCharm
    # @project：mDFA_Project

# -------------------------------

    # @filename：main.py
    # @teim：2025/11/1 11:05
    # @name：ShuaiFu Lu
    # @email：2301110293@pku.edu.cn

# -------------------------------
    
    
    
if __name__=='__main__':
    from arfima_ak import arfima_da
    from pymDFA import DFA_da,mDFA_fit_process_da
    X=arfima_da(n=15000,phi=[0.7],dfrac=0.15,size=10)
    S,Fs2=DFA_da(X)
    mDFA_fit_ARFIMA1d0=mDFA_fit_process_da(S,Fs2,pre_dict_path='Fs2p_ARFIMA(1,d,0)data.nc',
                                                 values=['d_value', 'a_value'], axis=0, loss='SR',
                                                 calculate_residuals=True,return_l=True,
                                                 standard_Fs2=True, return_Fs2=True)
    print(mDFA_fit_ARFIMA1d0)


