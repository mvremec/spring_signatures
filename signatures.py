# From NeuralHydrology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, signal
from xarray.core.dataarray import DataArray, Dataset
#from numba import njit
import matplotlib.colors as colors
from typing import Dict, List, Tuple, Optional, Union
from numpy import arange, diff, log, nan, sqrt
from pandas import DatetimeIndex, Series, Timedelta, cut
from scipy.stats import linregress
from datetime import datetime
from dateutil.relativedelta import relativedelta
import functools
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
import numpy as np
import pandas as pd
import copy
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf

# Set the minimum proportion of non-missing data required
def is_short(ts, limit=0.8):
    if len(ts.dropna())<len(ts)*limit:
        return True
    else:
        return False

# Compute mean 
def qmean(q, limit=0.8):
    if is_short(q, limit):
        return np.nan
    else:
        return np.nanmean(q)
        
# Compute median 
def qmedian(q, limit=0.8):
    if is_short(q, limit):
        return np.nan
    else:
        return np.nanmedian(q)
        
# Compute coefficient of variation    
def cv(q, limit=0.8):
    if is_short(q, limit):
        return np.nan
    else:
        return (np.nanstd(q)/np.nanmean(q)) * 100

# Compute standard deviation
def std(q, limit=0.8):
    if is_short(q, limit):
        return np.nan
    else:
        return np.nanstd(q)

# Compute Q5 
def q5(dis, limit=0.8):
    if is_short(dis, limit):
        return np.nan
    else:
        return np.nanquantile(dis, 0.05)

# Compute Q95 
def q95(dis, limit=0.8):
    if is_short(dis, limit):
        return np.nan
    else:
        return np.nanquantile(dis, 0.95)
# Compute Qmin
def qmin(dis, limit=0.8):
    if is_short(dis, limit):
        return np.nan
    else:
        return (dis).min()

# Compute Qmax
def qmax(dis, limit=0.8):
    if is_short(dis, limit):
        return np.nan
    else:
        return (dis).max()

# Compute slope FDC - from Pastas
def slope_fdc(da, lower_quantile=0.33, upper_quantile=0.66, limit=0.8):
    if is_short(da, limit):
        return np.nan
        # sort discharge by descending order
    else:
        fdc = da.sort_values(ascending=False).dropna()

        # get idx of lower and upper quantile
        idx_lower = np.round(lower_quantile * len(fdc)).astype(int)
        idx_upper = np.round(upper_quantile * len(fdc)).astype(int)
        value = (np.log(fdc[idx_lower] +
                        1e-8)) - np.log(fdc[idx_upper] + 1e-8) / (upper_quantile - lower_quantile)
        return value

# Compute DOY (in datetimeindex form - add .idxmax().dayofyear to get DOY) when half of annual flow goes through the hydrograph
def hfd_mean_annual(ts):
    if len(ts.dropna()) < len(ts) * 0.8:
        return np.nan
    else:
        ts=ts.interpolate().values
        hfd = abs(ts.cumsum()-ts.sum()/2).argmin()
        if hfd < 10:
            return np.nan
        else:
            return hfd

# Compute DOY (in datetimeindex form - add .idxmax().dayofyear to get DOY) when maximum occurs
def sftime(dis, limit=0.8):
    if is_short(dis, limit):
        return np.nan
    else:
        return dis.argmax()#.idxmax().dayofyear

# Compute DOY (in datetimeindex form - add .idxmax().dayofyear to get DOY) when minimum occurs
def qmin_time(dis, limit=0.8):
    if is_short(dis, limit):
        return np.nan
    else:
        return dis.argmin()#.idxmax().dayofyear
        
# Normalize time series       
def _normalize(series: Series) -> Series:
    series = (series - series.min()) / (series.max() - series.min())
    return series


# Compute Baseflow - from Pastas
def baseflow(series: Series, normalize: bool = True) -> Tuple[Series, Series]:
    """Baseflow function for the baseflow index and stability.
    """
    if normalize:
        series = _normalize(series)

    # A/B. Selecting minima hm over 5-day periods
    hm = series.resample("5D").min().dropna()

    # C. define the turning point ht (0.9 * head < adjacent heads)
    ht = pd.Series(dtype=float)
    for i, h in enumerate(hm.iloc[1:-1], start=1):
        if (h < hm.iloc[i - 1]) & (h < hm.iloc[i + 1]):
            ht[hm.index[i]] = h

    # ensure that index is a DatetimeIndex
    ht.index = pd.to_datetime(ht.index)

    # D. Interpolate
    ht = ht.resample("D").interpolate()

    # E. Assign a base head to each day
    ht[ht > series.resample("D").mean().loc[ht.index]] = series.resample("D").mean()

    return series, ht


# Compute Baseflowindex - from Pastas
def baseflow_index(series, normalize= True, limit=0.8):
    if is_short(series, limit):
        return np.nan
    else:
        series, ht = baseflow(series, normalize=normalize)
        if is_short(ht, limit):
            return np.nan
        else:
            ht_sum = ht.sum()
            if ht_sum == 0:  # check if ht_sum is zero
                return np.nan  # return a special value (np.nan in this case) when ht_sum is zero
            else:
                return series.resample("D").mean().sum() / ht_sum

# Compute BFI for shorter periods
def bfi_winter(series, ht, normalize=True, limit=0.8):
    if is_short(series, limit):
        return np.nan
    else:
        ht_sum = ht.sum()
        if ht_sum == 0:  # check if ht_sum is zero
            return np.nan  # return a special value (np.nan in this case) when ht_sum is zero
        else:
            return series.resample("D").mean().sum() / ht_sum

# Compute snowmelt onset  
def snowmelt_onset(flow_series, limit=0.8):
    if is_short(flow_series, limit):
        return np.nan
    else:
        # Calculate the mean flow
        mean_flow = flow_series.mean()

        # Calculate the cumulative departure series
        cumulative_departure = (flow_series - mean_flow).cumsum()

        # Find the index of the minimum cumulative departure
        onset_index = np.argmin(cumulative_departure.values)

        # Convert the index to the day of the year
        onset_day_of_year = flow_series.index[onset_index].dayofyear

        return onset_day_of_year

# Compute Parde coefficients
def parde_coefficients(ts, normalize=True):
    """Pastas
    """
    if normalize:
        series = _normalize(ts)
    coefficients = ts.groupby(series.index.month).mean() / ts.mean()
    coefficients.index.name = "month"
    return coefficients

# Compute seasonality Parde coefficient
def parde_seasonality(ts, normalize = True, limit=0.8):
    if is_short(ts, limit):
        return np.nan
    else:    
        coefficients = parde_coefficients(ts, normalize=normalize)
        return coefficients.max() - coefficients.min()    

# To DO_ ;MRC separation - test against Excel
def f_linear(x, a, b):
    return a * x + b 

def f_log(x, a, b):
    return a * np.log(x) + b

def f_exp(x, a, b):
    return b * np.exp(a*x)


def max_timing(ts):
    return ts1.idxmax().dayofyear

def alpha3(ts):
    mrc = mrc_trigonometry(ts, min_len=4)["values"]
    q10 = mrc.quantile(0.1)  # quantile of the recession
    q50 = mrc.quantile(0.5)  # quantile of the recession
    data_alpha3 = mrc[mrc<q10]
    popt3,pcov3 = curve_fit(f_exp, data_alpha3.index, data_alpha3.values, p0=[0.02, q50])
    return round(popt3[0], 5)

def mrc_separation(mrc_values, Q10=0.1, Q50=0.5, only_baseflow=False):
    if only_baseflow:
        q10 = mrc_values.quantile(0.1)  # quantile of the recession
        q50 = mrc_values.quantile(0.5)  # quantile of the recession
        data_alpha3 = mrc_values[mrc_values<q10]
        popt3,pcov3 = curve_fit(f_exp, data_alpha3.index, data_alpha3.values, p0=[0.02, q50])
        return popt3
    else:
        q10 = mrc_values.quantile(0.1)  # quantile of the recession
        q50 = mrc_values.quantile(0.5)  # quantile of the recession
        data_alpha3 = mrc_values[mrc_values<q10]
        data_alpha2 = mrc_values[(mrc_values>q10)&(mrc_values<q50)]
        data_alpha1 = mrc_values[mrc_values>q50]
        popt1,pcov1 = curve_fit(f_exp, data_alpha1.index, data_alpha1.values, p0=[0.04, q50])
        popt2,pcov2 = curve_fit(f_exp, data_alpha2.index, data_alpha2.values, p0=[0.02, q50])
        popt3,pcov3 = curve_fit(f_exp, data_alpha3.index, data_alpha3.values, p0=[0.02, q50])
        return [popt1, popt2, popt3]

def replace_consecutive_nans(series, n, value):
    count = 0
    for i in range(len(series)):
        if pd.isna(series[i]):
            count += 1
        else:
            count = 0
        if count >= n:
            series.loc[series.index[i-count+1:i+1]]
            count = 0
    return series

# Define custom function to replace duplicated consecutive NaN values
def replace_duplicated_consecutive_nans(series, value):
    s = series.copy()
    prev_value = None
    count = 0
    for i in range(len(series)):
        if pd.isna(series[i]):
            if count == 0:
                prev_value = series[i-1]
            count += 1
        else:
            if count > 0 and prev_value is not None and prev_value < series[i]:
                s.loc[s.index[i-count+1:i+1]] = value
            count = 0
    return s

def find_first_lower(arr, arg):
    lower_values = arr[arr < arg]
    if lower_values.size > 0:
        index = arr.index[np.where(arr == np.max(lower_values))[0][0]]
        return np.max(lower_values), index
    else:        
        return arg, arr.index[-1]
    
def find_first_higher(arr, arg):
    higher_values = arr[arr >= arg]
    if higher_values.size > 0:
        index = arr.index[np.where(arr == np.min(higher_values))[-1][-1]]
        return (np.min(higher_values), index)
    else:
        return arg, arr.index[-2]
    
def trigonometry(series_list):
    mrc0 = series_list[0]
    mrc = copy.deepcopy(series_list)
    for i in np.arange(0, len(mrc)-2):
        SECRECSEGLVALUE = np.max(mrc[i+1]["values"]) # Write the largest value in the next recession segment into variable SECRECSEGLVALUE
        FUPPVALUE, RELTFUPPVALUE = find_first_higher(mrc0["values"], SECRECSEGLVALUE) #Write the first higher value than SECRECSEGLVALUE in the previous recession segment into variable FUPPVALUE
        FDOWNVALUE, RELTFDOWNVALUE = find_first_lower(mrc0["values"], SECRECSEGLVALUE)
        if RELTFDOWNVALUE == RELTFUPPVALUE:
            RELTIMESHIFT = RELTFDOWNVALUE
        else:
            TGALPHA = ((FUPPVALUE - FDOWNVALUE) / (RELTFDOWNVALUE - RELTFUPPVALUE)) #Calculate tangens of the angle for the vertex defined by FUPPVALUE, FDOWNVALUE, RELTFUPPVALUE, RELTFDOWNVALUE
            if TGALPHA == 0:
                RELTIMESHIFT = RELTFDOWNVALUE
            else:    
                RELTIMESUR = ((SECRECSEGLVALUE - FDOWNVALUE) / TGALPHA) #Calcualte the surplus which needs to be subtracted from the RELTFDOWNVALUE in order to get relative time shift for the next recession segment
                RELTIMESHIFT = RELTFDOWNVALUE - RELTIMESUR
            #print(SECRECSEGLVALUE, FDOWNVALUE, TGALPHA)
        mrc[i+1].index = int(np.round(RELTIMESHIFT,0)) + mrc[i+1].index.copy().astype(int)
        mrc0 = pd.concat([mrc0, mrc[i+1]])
    return mrc0

def mrc_trigonometry(data, min_len=4):
    """Rewritten from VBA Excel code from Posavec et al. (2017)"""
    data3 = pd.DataFrame({"values":data, "rel":np.arange(0, len(data)), "date":data.index}, index=data.index)
    df=copy.deepcopy(data3)
    i = 2
    while True:
        if df.iloc[i, 0] >= df.iloc[i - 1, 0] and df.iloc[i, 0] <= df.iloc[i + 1, 0]:
            df = df.drop(df.index[i])
            i -= 1
            i += 1
        else:
            i += 1

        try:
            df.iloc[i + 1, 0]
        except IndexError:
            break

    if df.iloc[1, 0] < data3.iloc[2, 0]:
        df = df.drop(df.index[1])

    i = 2
    while True:
        try:
            if df.iloc[i, 0] > df.iloc[i - 1, 0] and pd.isna(df.iloc[i+1, 0]):
                df.iloc[i, 1:2] = np.nan
                i += 1
            else:
                i += 1
        except IndexError:
            break

        try:
            df.iloc[i, 0]
        except IndexError:
            break

    i = 1
    while True:
        if (df.iloc[i+1, 0] > df.iloc[i, 0]) or (df.iloc[i+1, 1] > (7 +  df.iloc[i, 1])):
            df.iloc[i+1:i+1, :] = df.iloc[i:i+1, :].shift(1)
            i += 2
        else:
            i += 1

        try:
            df.iloc[i+1, 0]
        except IndexError:
            break

    if pd.isna(df.iloc[0, 1]) and pd.isna(df.iloc[2, 1]):
        df = df.drop(df.index[[1, 2]])

    i = 1
    while i < len(df.index) - 1:
        if pd.isna(df.iloc[i-1, 1]) and pd.isna(df.iloc[i+1, 1]):
            df.drop(df.index[i:i+2], inplace=True)
        else:
            i += 1
        try:
            df.iloc[i+1, 0]
        except IndexError:
            break

    if pd.isna(df.iloc[0, 1]) and pd.isna(df.iloc[3, 1]):
        df = df.drop(df.index[[1, 3]])

    i = 1
    while i < len(df.index):
        if pd.isna(df.iloc[i - 1, 0]) and pd.isna(df.iloc[i + 2, 0]):
            df = df.drop(df.index[i:i + 3]).reset_index(drop=True)
        else:
            i += 1
        try:
            df.iloc[i+2, 0]
        except IndexError:
            break

    if pd.isna(df.iloc[0, 1]) and pd.isna(df.iloc[4, 1]):
        df = df.drop(df.index[1:4])

    i = 1
    while i < len(df.index):
        if pd.isna(df.iloc[i - 1, 0]) and pd.isna(df.iloc[i + 3, 0]):
            df = df.drop(df.index[i:i+4]).reset_index(drop=True)
        else:
            i += 1
        try:
            df.iloc[i+3, 0]
        except IndexError:
            break
    #print("Part1 DONE")
    # Segments sorting
    maksrpv = df["values"].max()
    df["maksrpv"] = maksrpv
    df.index = df["date"]
    df =  df.reindex(pd.date_range(data.index[0], data.index[-1]))
    # Call the custom function to replace consecutive NaNs with 9998
    df["values"] = replace_consecutive_nans(df["values"], 7, 9998)
    
    # Call the custom function to replace duplicated values
    df["values"] = replace_duplicated_consecutive_nans(df["values"], 9998)
    # Find indices where the series needs to be split
    split_indices = np.where((df["values"] == 9998) | (df["values"] == 9999))[0]
    # Initialize list to store segments
    segments = []

    # Iterate over split indices to extract segments
    for i in range(len(split_indices)):
        start_idx = 0 if i == 0 else split_indices[i-1] + 1
        end_idx = split_indices[i]
        segment = df.iloc[start_idx:end_idx,:]
        if 9998 not in segment.values and 9999 not in segment.values and not segment.dropna().empty:
            if len(segment["values"].dropna()) < min_len:
                pass
           # segments.append(segment.dropna())
            else:
                if (segment["values"].diff() > 0).any():
                    idx = (segment["values"][segment["values"].diff() > 0].index)[0]
                    if len(segment["values"].loc[:idx- pd.Timedelta(1, frequency="days")].dropna()) < 4:
                        pass
                    else:
                        segments.append(segment.loc[:idx- pd.Timedelta(1, frequency="days"), :].dropna())
                    if len(segment.loc[idx:, "values"].dropna()) < 4:
                        pass    
                    else:
                        segments.append(segment.loc[idx:,:].dropna())
                else:
                    segments.append(segment.dropna())

    # Add last segment after the last split index
    if len(split_indices) > 0 and split_indices[-1] < len(df["values"]) - 1:
        start_idx = split_indices[-1] + 1
        end_idx = len(df["values"])
        segment = df.iloc[start_idx:end_idx,:]
        if 9998 not in segment.values and 9999 not in segment.values and not segment.dropna().empty:
            if len(segment.dropna()) < min_len:
                pass
           # segments.append(segment.dropna())
            else:
                if (segment["values"].diff() > 0).any():
                    idx = (segment["values"][segment["values"].diff() > 0].index)[0]
                    if len(segment.loc[:idx- pd.Timedelta(1, frequency="days")].dropna()) < 4:
                        pass
                    else:
                        segments.append(segment.loc[:idx- pd.Timedelta(1, frequency="days"),:].dropna())
                    if len(segment.loc[idx:, "values"].dropna()) < 4:
                        pass    
                    else:
                        segments.append(segment.loc[idx:,:].dropna())
                else:
                    segments.append(segment.dropna())

    # Sort the list of DataFrames based on "values" column in descending order
    df_list_sorted = sorted(segments, key=lambda x: x['values'].max(), reverse=True)
    df_list_sorted1 = copy.deepcopy(df_list_sorted)
    dt = 0
    for i in np.arange(0, len(df_list_sorted)):
        df_list_sorted1[i]["date"] = df_list_sorted[i].index
        df_list_sorted1[i].index = np.arange(0, len(df_list_sorted[i].loc[:,"rel"]))# + dt
    return trigonometry(df_list_sorted1)
