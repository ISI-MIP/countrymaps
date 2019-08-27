#!/usr/bin/python



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
import pandas as pd
import re
import sys
sys.path.append('../')
#import counting_extremes as ce
from optparse import OptionParser

usage = 'usage: python %prog [options]'
parser = OptionParser(usage)
parser.add_option('-s', '--sec', action='store', type='string', dest='s', help='impact sector')
parser.add_option('-m', '--mod', action='store', type='string', dest='m', help='impact model')
parser.add_option('-g', '--gcm', action='store', type='string', dest='g', help='GCM')
parser.add_option('-c', '--cor', action='store', type='string', dest='c', help='country code or region name')
parser.add_option('-p', '--pop', action='store_true', dest='p', help='plot population fraction affected (default: plot area fraction affected)')
parser.add_option('-r', '--rel', action='store_true', dest='r', help='compute relative change (default: absolute change)')
(options, args) = parser.parse_args()
sec = options.s
mod = options.m
gcm = options.g
cor = options.c
lpopulation = options.p
lrelative = options.r
lregion = re.match('[A-Z][A-Z][A-Z]', cor) is None
lmultiplesectors = sec[:4] in ['sum+', 'max+']

exps = ['piControl', 'historical', 'rcp26', 'rcp60']
if cor == 'world':
    ofileformats = ['txt', 'npy'] if lmultiplesectors else ['pdf', 'txt', 'npy']
else:
    ofileformats = ['txt'] if lmultiplesectors else ['pdf', 'txt']

quantile_percentages = np.array([10., 20., 25., 40., 50., 60., 75., 80., 90.])
imedian = np.where(quantile_percentages == 50)[0][0]

figwidth, figheight = 10, 6
topmargin = .015
bottommargin = .08
leftmargin = .15
rightmargin = .005
alpha=0.25

# get piControl baseline temperature
ys, ye = ce.get_cmip5_experiment_period('piControl', gcm)
gmtpath = ce.gmtdir+gcm+'/tas_day_'+gcm+'_piControl_r1i1p1_EWEMBI_'+str(ys)+'-'+str(ye)+'.fldmean.yearmean.txt'
gmtpiControlmean = np.mean(np.loadtxt(gmtpath, skiprows=1, usecols=(1,)))

# get piControl expectation of AFA or PFA
relfix = '.relative' if lrelative else ''
secfix = '.multiplesectors' if lmultiplesectors else '.singlesector'
popfix = 'PFA' if lpopulation else 'AFA'
ivar = popfix+'_'+cor
idir = ce.odirCounting+sec+'/'+mod+'/'
pers = ce.get_isimip2b_period_names('piControl', 'IPSL-CM5A-LR')
piCmean = {}
piCse = {}
piCexp = {}
piAvail = {}
for per in pers:
    ys, ye = ce.isimip2b_period[per]
    nyears = ye - ys + 1
    piCexp[per] = np.empty(nyears, dtype=float)
    piCexp[per][:] = np.nan
    piCmean[per] = np.empty(nyears, dtype=float)
    piCmean[per][:] = np.nan
    piCse[per] = np.empty(nyears, dtype=float)
    piCse[per][:] = np.nan
    piAvail[per] = False

# set up figure
fig = plt.figure(figsize=(figwidth, figheight))
gs = gridspec.GridSpec(2, 2, left=leftmargin, right=1-rightmargin, bottom=bottommargin, top=1-topmargin, wspace=0, hspace=0)
try:
    region_adjective = ce.region_adjective[cor]
except:
    region_adjective = cor

# load and plot
pic_SEs_per_warming_level = [np.zeros(0) for warming_level in ce.warming_levels]
pic_data_per_warming_level = [np.zeros(0) for warming_level in ce.warming_levels]
data_per_warming_level = [np.zeros(0) for warming_level in ce.warming_levels]
data_per_warming_level_mean = [np.zeros(0) for warming_level in ce.warming_levels]
data_per_warming_level_nochange = [np.zeros(0) for warming_level in ce.warming_levels]
datapreviousyear_per_warming_level = [np.zeros(0) for warming_level in ce.warming_levels]
deltagmts_per_warming_level = [np.zeros(0) for warming_level in ce.warming_levels]
years_per_warming_level = [np.zeros(0, dtype=int) for warming_level in ce.warming_levels]
exps_per_warming_level = [np.zeros(0, dtype=str) for warming_level in ce.warming_levels]
nge_per_warming_level = np.zeros(ce.warming_levels.size)
nlt_per_warming_level = np.zeros_like(nge_per_warming_level)
lower_limit_per_warming_level = np.empty_like(nge_per_warming_level); lower_limit_per_warming_level[:] = np.nan
upper_limit_per_warming_level = np.empty_like(nge_per_warming_level); upper_limit_per_warming_level[:] = np.nan
p01pic, p50pic, p99pic, meanpic, sepic, p01fut, p50fut, p99fut, meanfut, sefut, stdfut = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
hist0prev, rcpn0prev, rcpe0prev = np.nan, np.nan, np.nan
dxdy = {'dx': np.zeros(0), 'dy': np.zeros(0)}
ts_future = None
pic_ts = {}
for exp in exps:
    color = ce.cmip5_experiment_color[exp]
 
    # load experiment temperatures
    ysexp, yeexp = ce.get_cmip5_experiment_period(exp, gcm)
    gmtpath = ce.gmtdir+gcm+'/tas_day_'+gcm+'_'+exp+'_r1i1p1_EWEMBI_'+str(ysexp)+'-'+str(yeexp)+'.fldmean.yearmean.txt'
    gmtallperiods = np.loadtxt(gmtpath, skiprows=1, usecols=(1,))

    pers = ce.get_isimip2b_period_names(exp, gcm)
    if 'historical' in pers:  # move historical to the end for proper computation of piControl percentiles
        pers.remove('historical')
        pers.append('historical')
    for p, per in enumerate(pers):
        ys, ye = ce.isimip2b_period[per]
        years = np.arange(ys, ye+1)
        deltagmt = gmtallperiods[ys-ysexp:ye-ysexp+1] - gmtpiControlmean

        # load anomalies
        ifile = ce.ifileprefix_countryregionlevel(sec, mod, gcm, exp, per, lregion, lpopulation, False)+'.'+ce.ncs
        ts = ce.loadncfile(idir+ifile, ivar)
        if ts is None:
            if exp == 'piControl': pic_ts[per] = None
            continue
        ts = 100. * ts[:,0,0]

        if exp == 'piControl':  # get piControl percentiles
            pic_ts[per] = ts
            if per == 'pre-industrial':
                piAvail[per] = True
                p01pic = np.percentile(ts, 2.)
                p50pic = np.percentile(ts, 50.)
                p99pic = np.percentile(ts, 98.)
                meanpic = np.mean(ts)
                sepic = np.std(ts, ddof=1) / np.sqrt(ts.size)
                piCexp[per][:] = p50pic
                piCmean[per][:] = meanpic
                piCse[per][:] = sepic
                hist0prev = ts[-1]
            elif per == 'future':
                piAvail[per] = True
                p01fut = np.percentile(ts, 2.)
                p50fut = np.percentile(ts, 50.)
                p99fut = np.percentile(ts, 98.)
                meanfut = np.mean(ts)
                stdfut = np.std(ts, ddof=1)
                sefut = stdfut / np.sqrt(ts.size)
                piCexp[per][:] = p50fut
                piCexp['future_extended'][:] = p50fut
                piCmean[per][:] = meanfut
                piCmean['future_extended'][:] = meanfut
                piCse[per][:] = sefut
                piCse['future_extended'][:] = sefut
                ts_future = ts.copy()
            elif per == 'future_extended':
                piAvail[per] = True
                if piAvail['future']:
                    p01fut = np.percentile(np.concatenate((ts, ts_future)), 2.) 
                    p50fut = np.percentile(np.concatenate((ts, ts_future)), 50.) 
                    p99fut = np.percentile(np.concatenate((ts, ts_future)), 98.) 
                    meanfut = np.mean(np.concatenate((ts, ts_future))) 
                    stdfut = np.std(np.concatenate((ts, ts_future)), ddof=1)
                    sefut = stdfut / np.sqrt(ts.size + ts_future.size)
                else:
                    p01fut = np.percentile(ts, 2.)
                    p50fut = np.percentile(ts, 50.)
                    p99fut = np.percentile(ts, 98.)
                    meanfut = np.mean(ts)
                    stdfut = np.std(ts, ddof=1)
                    sefut = stdfut / np.sqrt(ts.size)
                piCexp[per][:] = p50fut
                piCexp['future'][:] = p50fut
                piCmean[per][:] = meanfut
                piCmean['future'][:] = meanfut
                piCse[per][:] = sefut
                piCse['future'][:] = sefut
            elif per == 'historical':  # check if there is a jump in the piControl time series from 1680 to 1861
                piAvail[per] = True
                if piAvail['pre-industrial']:
                    p01his = None
                    p50his = None
                    p99his = None
                    meanhis = None
                    sehis = None
                    piCexp[per][:] = p50pic + np.linspace(0, 1, ye-ys+2, endpoint=True)[1:] * (p50fut - p50pic)
                    piCmean[per][:] = meanpic + np.linspace(0, 1, ye-ys+2, endpoint=True)[1:] * (meanfut - meanpic)
                    piCse[per][:] = sepic + np.linspace(0, 1, ye-ys+2, endpoint=True)[1:] * (sefut - sepic)
                else:
                    ts50 = ts[:50]
                    p01his = np.percentile(ts50, 2.)
                    p50his = np.percentile(ts50, 50.)
                    p99his = np.percentile(ts50, 98.)
                    meanhis = np.mean(ts50)
                    sehis = np.std(ts50, ddof=1) / np.sqrt(50)
                    piCexp[per][:] = p50his + np.linspace(0, 1, ye-ys+2, endpoint=True)[1:] * (p50fut - p50his)
                    piCmean[per][:] = meanhis + np.linspace(0, 1, ye-ys+2, endpoint=True)[1:] * (meanfut - meanhis)
                    piCse[per][:] = sehis + np.linspace(0, 1, ye-ys+2, endpoint=True)[1:] * (sefut - sehis)

        else:  # collect data per warming level and for detection level quantile regression
            if exp == 'historical':
                rcpn0prev = ts[-1]
                if mod == 'CLM45' and sec == 'driedarea':  # work around the 2005soc problem that only pertains to CLM45
                    ts50 = ts[:50]  # compute piControl stats exclusively from first 50 years of historical run
                    # overwrite everything that has been calculated before
                    p01his = np.percentile(ts50, 2.); p01pic = p01his; p01fut = p01his
                    p50his = np.percentile(ts50, 50.); p50pic = p50his; p50fut = p50his
                    p99his = np.percentile(ts50, 98.); p99pic = p99his; p99fut = p99his
                    meanhis = np.mean(ts50); meanpic = meanhis; meanfut = meanhis
                    stdhis = np.std(ts50, ddof=1); stdpic = stdhis; stdfut = stdhis
                    sehis = stdhis / np.sqrt(50); sepic = sehis; sefut = sehis
                    for ppeerr in ce.get_isimip2b_period_names('piControl', gcm):
                        piCexp[ppeerr][:] = p50his
                        piCmean[ppeerr][:] = meanhis
                        piCse[ppeerr][:] = sehis
            elif per == 'future': rcpe0prev = ts[-1]
            ts_change = 100 * (ts - piCexp[per]) / piCexp[per] if lrelative else ts - piCexp[per]
            ts_change_lower_limit = 100 * (0 - piCexp[per]) / piCexp[per] if lrelative else 0 - piCexp[per]
            ts_change_upper_limit = 100 * (100 - piCexp[per]) / piCexp[per] if lrelative else 100 - piCexp[per]
            ts_change_mean = 100 * (ts - piCmean[per]) / piCmean[per] if lrelative else ts - piCmean[per]
            for w, warming_level in enumerate(ce.warming_levels):
                llt = np.logical_and(deltagmt >= warming_level - ce.delta_warming_level, deltagmt < warming_level)
                lge = np.logical_and(deltagmt >= warming_level, deltagmt < warming_level + ce.delta_warming_level)
                nlt_per_warming_level[w] = nlt_per_warming_level[w] + llt.sum()
                nge_per_warming_level[w] = nge_per_warming_level[w] + lge.sum()
                lltorge = np.logical_or(llt, lge)
                if lltorge.sum():
                    iprevious = np.where(lltorge)[0] - 1
                    ts_change_previous = ts_change[iprevious]
                    if np.any(iprevious == -1):
                        ts_change_previous[0] = hist0prev if exp == 'historical' else rcpn0prev if per == 'future' else rcpe0prev
                    lower_limit = min(ts_change_lower_limit[lltorge])
                    lower_limit_per_warming_level[w] = lower_limit if np.isnan(lower_limit_per_warming_level[w]) else min(lower_limit_per_warming_level[w], lower_limit)
                    upper_limit = max(ts_change_upper_limit[lltorge])
                    upper_limit_per_warming_level[w] = upper_limit if np.isnan(upper_limit_per_warming_level[w]) else max(upper_limit_per_warming_level[w], upper_limit)
                    data_per_warming_level[w] = np.concatenate((data_per_warming_level[w], ts_change[lltorge]))
                    datapreviousyear_per_warming_level[w] = np.concatenate((datapreviousyear_per_warming_level[w], ts_change_previous))
                    deltagmts_per_warming_level[w] = np.concatenate((deltagmts_per_warming_level[w], deltagmt[lltorge]))
                    years_per_warming_level[w] = np.concatenate((years_per_warming_level[w], years[lltorge]))
                    exps_per_warming_level[w] = np.concatenate((exps_per_warming_level[w], np.array([exp for l in np.arange(llt.sum()+lge.sum())])))
                    pic_data_per_warming_level[w] = np.concatenate((pic_data_per_warming_level[w], pic_ts[per][lltorge]))  # for quantification of pure effect of socio-economic development
                    pic_SEs_per_warming_level[w] = np.concatenate((pic_SEs_per_warming_level[w], piCse[per][lltorge]))
                    data_per_warming_level_mean[w] = np.concatenate((data_per_warming_level_mean[w], ts_change_mean[lltorge]))
                    data_per_warming_level_nochange[w] = np.concatenate((data_per_warming_level_nochange[w], ts[lltorge]))
                dxdy['dx'] = np.concatenate((dxdy['dx'], deltagmt))
                dxdy['dy'] = np.concatenate((dxdy['dy'], ts_change))

        # plot versus time and temperature change
        ax = plt.subplot(gs[0,0])
        plt.scatter(years, ts, color=color)
        ax = plt.subplot(gs[0,1])
        plt.scatter(deltagmt, ts, color=color, label=None if p else exp)
        ax = plt.subplot(gs[1,0])
        if exp != 'piControl': plt.scatter(years, ts_change, color=color)
        ax = plt.subplot(gs[1,1])
        if exp != 'piControl': plt.scatter(deltagmt, ts_change, color=color)

# plot piControl percentiles
ax = plt.subplot(gs[0,0])
pers = ce.get_isimip2b_period_names('piControl', gcm)
for p, per in enumerate(pers):
    if per == 'future' and not piAvail[per] and not piAvail['future_extended']: continue
    elif not piAvail[per]: continue
    ys, ye = ce.isimip2b_period[per]
#    plt.plot(np.arange(ys, ye+1), piCexp[per], color='0.5', label=None if p else 'piControl median and 96% range')
#    p01 = [p01pic, p01pic] if per == 'pre-industrial' else [p01fut, p01fut] if per[:6] == 'future' else [p01pic, p01fut] if p01his is None else [p01his, p01fut]
#    p99 = [p99pic, p99pic] if per == 'pre-industrial' else [p99fut, p99fut] if per[:6] == 'future' else [p99pic, p99fut] if p01his is None else [p99his, p99fut]
#    plt.fill_between([ys-.5, ye+.5], p01, p99, color='0.85', edgecolor='none', zorder=-1)
    plt.plot(np.arange(ys, ye+1), piCmean[per], color='0.5', label=None if p else 'piControl mean')

# prettify plots versus time
ax = plt.subplot(gs[0,0])
ax.set_xlim(1660, 2300)
ax.set_xticklabels([])
ax.set_yticks(ax.get_yticks()[1:-1])
plt.ylabel(region_adjective+(' Population' if lpopulation else ' Land Area')+' Fraction\nAnnually '+('Exposed to\n' if lpopulation else 'Affected by\n')+ce.hazard_long_name[sec]+' [%]')
ax.annotate('A', xy=(.02,.97), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontweight='bold')
ylimtop = ax.get_ylim()
ax = plt.subplot(gs[1,0])
ax.set_xlim(1660, 2300)
plt.axhline(y=0, color=ce.cmip5_experiment_color['piControl'], zorder=0)
plt.xlabel('Year')
plt.ylabel(('Relative' if lrelative else 'Absolute')+' Change in '+region_adjective+'\n'+('Population' if lpopulation else 'Land Area')+' Fraction Annually\n'+('Exposed to ' if lpopulation else 'Affected by ')+ce.hazard_long_name[sec]+' [%]')
ax.set_yticks(ax.get_yticks()[1:-1])
ax.set_xticks(ax.get_xticks()[1:-1])
ax.annotate('C', xy=(.02,.97), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontweight='bold')
ylimbottom = ax.get_ylim()

# prettify plot versus temperature change
ax = plt.subplot(gs[0,1])
ax.set_xlim(-0.5, 4.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.legend(loc='lower right', fontsize='medium', frameon=False, labelspacing=.1, handletextpad=.4, numpoints=1, handlelength=1.3, ncol=1)
ax.annotate('B', xy=(.02,.97), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontweight='bold')
ax.set_ylim(ylimtop)
ax = plt.subplot(gs[1,1])
ax.set_xlim(-0.5, 4.5)
plt.axhline(y=0, color=ce.cmip5_experiment_color['piControl'], zorder=0)
ax.set_xticks(ax.get_xticks()[1:-1])
ax.set_yticklabels([])
plt.xlabel(u'\u2206GMT [\u00B0C]')
ax.annotate('D', xy=(.02,.97), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontweight='bold')

# make quantile regression for 80th percentile of change versus deltagmt and determine detection level
ldiscard = np.logical_or(np.logical_or(nlt_per_warming_level == 0, nge_per_warming_level == 0), nlt_per_warming_level + nge_per_warming_level < 5)
if np.all(ldiscard) or not np.any(np.array(piAvail.values())) or np.all(np.isnan(dxdy['dy'])) or not np.any(np.nonzero(dxdy['dy'])):
    detection_level = np.nan
else:
    piCexpfut = piCexp['future'][0]
    p01_change = 100 * (p01fut - piCexpfut) / piCexpfut if lrelative else p01fut - piCexpfut
    p99_change = 100 * (p99fut - piCexpfut) / piCexpfut if lrelative else p99fut - piCexpfut
#    plt.fill_between([-0.5, 4.5], p01_change, p99_change, color='0.85', edgecolor='none', zorder=-2)
    try:
        if sec in ['heatwavedarea', 'max+driedarea+heatwavedarea', 'sum+driedarea+heatwavedarea']:  # limit data that is used for regression to GMT change window [0, 1.5]
            ikeep = np.logical_and(dxdy['dx'] > 0, dxdy['dx'] < 1.5)
            dxdy['dy'] = dxdy['dy'][ikeep]
            dxdy['dx'] = dxdy['dx'][ikeep]
        qr = smf.quantreg('dy ~ dx', pd.DataFrame(data=dxdy))
        q4regression = qr.fit(q=.8).predict({'dx': [0, 1]})
        detection_level = (p99_change - q4regression[0])/( q4regression[1] -  q4regression[0])
    except ValueError:  # dont know why this still sometimes happens yet it does
        print ('smf.quantreg results in ValueError !!! setting detection_level to NaN ...')
        detection_level = np.nan
    #plt.axvline(x=detection_level, color='green', zorder=0)

# compute and plot quantiles per warming level
lower_limit_per_warming_level[ldiscard] = np.nan
upper_limit_per_warming_level[ldiscard] = np.nan
quantiles_per_warming_level = np.empty((quantile_percentages.size, ce.warming_levels.size), dtype=float)
quantiles_per_warming_level[:] = np.nan
# make a piecewise linear quantile regression and save middle points
for w, warming_level in enumerate(ce.warming_levels):
    if ldiscard[w]: continue
    if warming_level > 3:  # for some reason (presumably too few data points) the QR does not work at 3.5 and 4 degrees global warming
        for q, quantile_percentage in enumerate(quantile_percentages):
            quantiles_per_warming_level[q,w] = np.percentile(data_per_warming_level[w], quantile_percentage)
        continue
    dxdy = {'dx': deltagmts_per_warming_level[w], 'dy': data_per_warming_level[w]}
    try:
        qr = smf.quantreg('dy ~ dx', pd.DataFrame(data=dxdy))
    except ValueError:  # dxdy sometimes contain NaNs or divide by zero producing infs occur during quantile regression
        for q, quantile_percentage in enumerate(quantile_percentages):
            quantiles_per_warming_level[q,w] = np.percentile(data_per_warming_level[w], quantile_percentage)
        continue
    for q, quantile_percentage in enumerate(quantile_percentages):
        try:
            quantiles_per_warming_level[q,w] = qr.fit(q=.01*quantile_percentage).predict({'dx': [warming_level]})
        except ValueError:
            quantiles_per_warming_level[q,w] = np.percentile(data_per_warming_level[w], quantile_percentage)
    # make sure there are no nans or infs and that quantiles and limits are consistent ...
    if np.any(np.isnan(quantiles_per_warming_level[:,w])) or np.any(np.isinf(quantiles_per_warming_level[:,w])) or np.any(quantiles_per_warming_level[:,w] < lower_limit_per_warming_level[w]) or np.any(quantiles_per_warming_level[:,w] > upper_limit_per_warming_level[w]) or np.any(np.diff(quantiles_per_warming_level[:,w]) <= 0):
        for q, quantile_percentage in enumerate(quantile_percentages):
            quantiles_per_warming_level[q,w] = np.percentile(data_per_warming_level[w], quantile_percentage)
if 'VISIT' in mod or 'ORCHIDEE' in mod:  # interpolate linearly over data-sparse deltagmt values
    i_deltagmt_stable = 4 if gcm == 'IPSL-CM5A-LR' else 3 if gcm == 'MIROC5' else 2 if gcm == 'GFDL-ESM2M' else 0
    offset0 = -quantiles_per_warming_level[imedian,0]
    for w, warming_level in enumerate(ce.warming_levels[1:i_deltagmt_stable]): 
        if ldiscard[w+1]: continue
        weight1 = (w+1.) / i_deltagmt_stable
        for q, quantile_percentage in enumerate(quantile_percentages):
            quantiles_per_warming_level[q,w+1] = (1.-weight1) * (quantiles_per_warming_level[q,0] + offset0) + weight1 * quantiles_per_warming_level[q,i_deltagmt_stable]
#plt.plot(ce.warming_levels, quantiles_per_warming_level[4], color='red')
#plt.fill_between(ce.warming_levels, quantiles_per_warming_level[1], quantiles_per_warming_level[-2], color='red', edgecolor='none', alpha=alpha, zorder=-1)
#ax.set_ylim(ylimbottom)

# compute piControl quantiles per warming level for quantification of pure effect of socio-economic development
pic_quantiles_per_warming_level = np.empty((quantile_percentages.size, ce.warming_levels.size), dtype=float)
pic_quantiles_per_warming_level[:] = np.nan
# make a piecewise linear quantile regression and save middle points
for w, warming_level in enumerate(ce.warming_levels):
    if ldiscard[w]: continue
    if warming_level > 3:  # for some reason (presumably too few data points) the QR does not work at 3.5 and 4 degrees global warming
        for q, quantile_percentage in enumerate(quantile_percentages):
            pic_quantiles_per_warming_level[q,w] = np.percentile(pic_data_per_warming_level[w], quantile_percentage)
        continue
    dxdy = {'dx': deltagmts_per_warming_level[w], 'dy': pic_data_per_warming_level[w]}
    try:
        qr = smf.quantreg('dy ~ dx', pd.DataFrame(data=dxdy))
    except ValueError:  # dxdy sometimes contain NaNs or divide by zero producing infs occur during quantile regression
        for q, quantile_percentage in enumerate(quantile_percentages):
            pic_quantiles_per_warming_level[q,w] = np.percentile(pic_data_per_warming_level[w], quantile_percentage)
        continue
    for q, quantile_percentage in enumerate(quantile_percentages):
        try:
            pic_quantiles_per_warming_level[q,w] = qr.fit(q=.01*quantile_percentage).predict({'dx': [warming_level]})
        except ValueError:
            pic_quantiles_per_warming_level[q,w] = np.percentile(pic_data_per_warming_level[w], quantile_percentage)
    # make sure there are no nans or infs and that quantiles are consistent ...
    if np.any(np.isnan(pic_quantiles_per_warming_level[:,w])) or np.any(np.isinf(pic_quantiles_per_warming_level[:,w])) or np.any(np.diff(pic_quantiles_per_warming_level[:,w]) <= 0):
        for q, quantile_percentage in enumerate(quantile_percentages):
            pic_quantiles_per_warming_level[q,w] = np.percentile(pic_data_per_warming_level[w], quantile_percentage)

# compute mean changes, their standard errors and natural variability per warming level
means_per_warming_level = np.empty(ce.warming_levels.size, dtype=float)
means_per_warming_level[:] = np.nan
stds_per_warming_level = means_per_warming_level.copy()
SEmeans_per_warming_level = means_per_warming_level.copy()
means_per_warming_level_nochange = means_per_warming_level.copy()
stds_per_warming_level_nochange = means_per_warming_level.copy()
for w, warming_level in enumerate(ce.warming_levels):
    if ldiscard[w]: continue
    means_per_warming_level_nochange[w] = np.mean(data_per_warming_level_nochange[w])
    stds_per_warming_level_nochange[w] = np.std(data_per_warming_level_nochange[w], ddof=1)
    means_per_warming_level[w] = np.mean(data_per_warming_level_mean[w])
    stds_per_warming_level[w] = np.std(data_per_warming_level_mean[w], ddof=1)
    SEmeans_per_warming_level[w] = np.sqrt(np.mean(np.square(pic_SEs_per_warming_level[w])) + np.square(stds_per_warming_level[w]) / data_per_warming_level_mean[w].size)
if 'VISIT' in mod or 'ORCHIDEE' in mod:  # interpolate linearly over data-sparse deltagmt values
    i_deltagmt_stable = 4 if gcm == 'IPSL-CM5A-LR' else 3 if gcm == 'MIROC5' else 2 if gcm == 'GFDL-ESM2M' else 0
    offset0 = -means_per_warming_level[0]
    for w, warming_level in enumerate(ce.warming_levels[1:i_deltagmt_stable]): 
        if ldiscard[w+1]: continue
        weight1 = (w+1.) / i_deltagmt_stable
        means_per_warming_level[w+1] = (1.-weight1) * (means_per_warming_level[0] + offset0) + weight1 * means_per_warming_level[i_deltagmt_stable]

# plot mean changes and year-to-year variability per warming level
mpwl2plot = means_per_warming_level.copy()
mpwl2plot[0] = 0
plt.plot(ce.warming_levels, mpwl2plot, color='red')
plt.fill_between(ce.warming_levels, mpwl2plot-stds_per_warming_level, mpwl2plot+stds_per_warming_level, color='red', edgecolor='none', alpha=alpha, zorder=-1)
ax.set_ylim(ylimbottom)

# save plot and warming level stats
odir = popfix+'.countryregionlevel.versus.gmt.and.timeseries'+secfix+relfix
ofile = popfix+relfix.replace('.', '_')+'_'+sec+'_'+mod+'_'+gcm+'_'+cor
pfile = 'piControl_'+ofile
nfile = 'nochange_'+ofile
for fmt in ofileformats:
    if fmt != 'pdf' or np.any(np.nonzero(nlt_per_warming_level)) or np.any(np.nonzero(nge_per_warming_level)):
        opath = fmt+'/'+odir+'/'+ofile+'.'+fmt
        ppath = fmt+'/'+odir+'/'+pfile+'.'+fmt
        npath = fmt+'/'+odir+'/'+nfile+'.'+fmt
        ce.ensure_dir(opath)
        if fmt == 'txt': 
            # save scenario warming level stats
            warming_level_stats = np.stack((ce.warming_levels,)+tuple([quantiles_per_warming_level[q] for q, quantile_percentage in enumerate(quantile_percentages)])+(lower_limit_per_warming_level, upper_limit_per_warming_level, nlt_per_warming_level, nge_per_warming_level, means_per_warming_level, stds_per_warming_level, SEmeans_per_warming_level), axis=-1)
            np.savetxt(opath, warming_level_stats, 
                       header='detection_level_K %.5f piControl_2005soc_mean %.5f piControl_2005soc_std %.5f piControl_2005soc_median %.5f\n'%(detection_level, meanfut, stdfut, p50fut)+'warming_level_K '+' '.join(['quintile_%.0f_percent'%q for q in quantile_percentages])+' lower_limit_percent upper_limit_percent n_data_points_lt_warming_level n_data_points_ge_warming_level mean_percent std_percent SE_mean_percent',
                       fmt='%3.1f '+' '.join(['%.5f' for q in quantile_percentages])+' %.5f %.5f %.0f %.0f %.5f %.5f %.5f')
            # save piControl warming level stats
            warming_level_stats = np.stack((ce.warming_levels,)+tuple([pic_quantiles_per_warming_level[q] for q, quantile_percentage in enumerate(quantile_percentages)]), axis=-1)
            np.savetxt(ppath, warming_level_stats, 
                       header='piControl_1860soc_median %.5e\n'%(piCexp['historical'][0])+'warming_level_K '+' '.join(['quintile_%.0f_percent'%q for q in quantile_percentages]),
                       fmt='%3.1f '+' '.join(['%.5e' for q in quantile_percentages]))
            # save nochange warming level stats
            warming_level_stats = np.stack((ce.warming_levels, means_per_warming_level_nochange, stds_per_warming_level_nochange), axis=-1)
            np.savetxt(npath, warming_level_stats, header='For ISIpedia data cube', fmt='%3.1f %.5e %.5e')
        elif fmt == 'npy':
            for w, warming_level in enumerate(ce.warming_levels):
                if np.ma.isMaskedArray(data_per_warming_level[w]):
                    np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.data.'%(warming_level)+fmt, data_per_warming_level[w].data)
                else:
                    np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.data.'%(warming_level)+fmt, data_per_warming_level[w])
                if np.ma.isMaskedArray(datapreviousyear_per_warming_level[w]):
                    np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.datapreviousyear.'%(warming_level)+fmt, datapreviousyear_per_warming_level[w].data)
                else:
                    np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.datapreviousyear.'%(warming_level)+fmt, datapreviousyear_per_warming_level[w])
                np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.deltagmts.'%(warming_level)+fmt, deltagmts_per_warming_level[w])
                np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.years.'%(warming_level)+fmt, years_per_warming_level[w])
                np.save(fmt+'/'+odir+'/'+ofile+'_%3.1f.exps.'%(warming_level)+fmt, exps_per_warming_level[w])
        else:
            plt.savefig(opath)


#%%