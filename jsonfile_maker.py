# To: generate country JSON files for ISIpedia project
#       1. extract country information
#       2. manupurate country stats
#       3. output them in JSON
# By: Yusuke Satoh (IIASA)
# On: 2018.06.01 version1
#     2019.07.15 version2
# Note: This code does not pay attention to memory efficiency so much...  (readability > speed > memory...)
#-----------------------------------------------------------------------------------------------------------------------
import sys
import os
import re
import time
#import json
import simplejson as json
import datetime
import numpy as np
from tqdm import tqdm
from itertools import product
from netCDF4 import Dataset
from numpy import array, resize, where, isinf, isnan, median, percentile
from numpy.ma import masked_array, masked_equal, masked_less, masked_less_equal, masked_greater_equal, masked_greater
from os.path import join as genpath


# Select these parameters ----------------------------------------------------------------------------------------------
impact_indices = [
    #'waterscarcity',  # water
    'FM',  # water
                  ]

JSONOUTLOUD = True
WHO_ARE_YOU = 'Yusuke Satoh (IIASA)'


# Edit hear for your environment... -----------------------------------------------------------------------------------
DATA_DIR_MAIN = '/home/satoh/sraid02/data/isipedia/sectoral_indicators'

OUTPUT_DIR = genpath('/home/satoh/hraid05/fig/isipedia', 'data_cube')

COUNTRYMAP_DIR = '/home/satoh/hraid05/data/mapmask/isipedia'
COUNTRYMAP_PATH = genpath(COUNTRYMAP_DIR, 'CountryMask.NtoS.plusATA.nc')
COUNTRY_List_PATH = genpath(COUNTRYMAP_DIR, 'cc_cl.txt')

METRIC_INPUT_DIR = '/home/satoh/hraid05/data/isimip2b/in'

AREAPATH = '/home/satoh/hraid05/data/grd_ara.LE.hlf'

STEFAN_FILE_DIR = '/home/satoh/hraid05/data/isipedia/isipedia_data_given_by_Julian.20190612/stefan_temperature_change'


# Basically, you don't need to touch these -----------------------------------------------------------------------------
code_version = 2.0

# (period, scenario)
periods_scenarios = [('pre-industrial', 'piControl'), ('historical', 'historical'), 
                     ('future', 'rcp26'), ('future_extended', 'rcp26'), 
                     ('future', 'rcp60'), ('future_extended', 'rcp60'),
                     ]
scenarios_all = list(dict.fromkeys([scenario for period, scenario in periods_scenarios]))
scenarios_rcp = [scenario for period, scenario in periods_scenarios if period == 'future']

assessment_category  = 'ISIMIP-projections'
temporal_types       = ['timeslices', 'temperature-change'] 
output_types         = ['absolute', 'absolute-changes', 'relative-changes'] 
ensemble_stats_types = ['median', 'interannual_standard_deviation', 'shading_lower_border', 'shading_upper_border', 'runs']
run_stats_types      = ['mean',   'interannual_standard_deviation', 'shading_lower_border', 'shading_upper_border']
items                = ['pop', 'area']  #,'index']
years_full           = range(1661, 2299+1)

dict_prds = {
    'timeslices': [(1661, 1680), (1681, 1700), (1701, 1720), (1721, 1740), (1741, 1760), (1761, 1780), (1781, 1800), (1801, 1820), (1821, 1840), (1841, 1860),  # pre-industrial
                   (1861, 1880), (1881, 1900), (1901, 1920), (1921, 1940), (1941, 1960), (1961, 1980), (1981, 2000),                                            # hisitorical
                   (2001, 2020),                                                                                                                                # need historical and future
                   (2021, 2040), (2041, 2060), (2061, 2080),                                                                                                    # future
                   (2081, 2100),                                                                                                                                # need future and future_extended
                   (2101, 2120), (2121, 2140), (2141, 2160), (2161, 2180), (2181, 2200), (2201, 2220), (2221, 2240), (2241, 2260), (2261, 2280)],               # future_extended
    'temperature-change': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0]
    }

# the first years of periods...
dict_timing = {
    'timeslices': {(1661, 1680): (1661, 1680), (1681, 1700): (1681, 1700),
                   (1701, 1720): (1701, 1720), (1721, 1740): (1721, 1740), (1741, 1760): (1741, 1760), (1761, 1780): (1761, 1780), (1781, 1800): (1781, 1800),
                   (1801, 1820): (1801, 1820), (1821, 1840): (1821, 1840), (1841, 1860): (1841, 1860), (1861, 1880): (1861, 1880), (1881, 1900): (1881, 1900),
                   (1901, 1920): (1901, 1920), (1921, 1940): (1921, 1940), (1941, 1960): (1941, 1960), (1961, 1980): (1961, 1980), (1981, 2000): (1981, 2000),
                   (2001, 2020): (2001, 2020), (2021, 2040): (2021, 2040), (2041, 2060): (2041, 2060), (2061, 2080): (2061, 2080), (2081, 2100): (2081, 2100),
                   (2101, 2120): (2101, 2120), (2121, 2140): (2121, 2140), (2141, 2160): (2141, 2160), (2161, 2180): (2161, 2180), (2181, 2200): (2181, 2200),
                   (2201, 2220): (2201, 2220), (2221, 2240): (2221, 2240), (2241, 2260): (2241, 2260), (2261, 2280): (2261, 2280)},
    'temperature-change': {   # TODO: 0 degree???   the first year of 31 year window
            'rcp26': {'HadGEM2-ES':   {0: 1661, 0.5: 1985, 1: 1997, 1.5: 2011, 2: None, 2.5: None, 3: None, 3.5: None, 4:None},
                      'IPSL-CM5A-LR': {0: 1661, 0.5: 1946, 1: 1978, 1.5: 1994, 2: 2014, 2.5: None, 3: None, 3.5: None, 4:None},
                      'GFDL-ESM2M':   {0: 1661, 0.5: 1974, 1: 1999, 1.5: None, 2: None, 2.5: None, 3: None, 3.5: None, 4:None},
                      'MIROC5':       {0: 1661, 0.5: 1981, 1: 2000, 1.5: 2033, 2: None, 2.5: None, 3: None, 3.5: None, 4:None},},
            'rcp45': {'HadGEM2-ES':   {0: 1661, 0.5: 1985, 1: 2000, 1.5: 2016, 2: 2031, 2.5: 2047, 3: None, 3.5: None, 4:None},
                      'IPSL-CM5A-LR': {0: 1661, 0.5: 1946, 1: 1978, 1.5: 1996, 2: 2014, 2.5: 2029, 3: 2051, 3.5: 2130, 4:None},
                      'GFDL-ESM2M':   {0: 1661, 0.5: 1974, 1: 1999, 1.5: 2034, 2: None, 2.5: None, 3: None, 3.5: None, 4:None},
                      'MIROC5':       {0: 1661, 0.5: 1980, 1: 2002, 1.5: 2024, 2: 2054, 2.5: None, 3: None, 3.5: None, 4:None},},
            'rcp60': {'HadGEM2-ES':   {0: 1661, 0.5: 1984, 1: 1998, 1.5: 2017, 2: 2035, 2.5: 2048, 3: 2061, 3.5: None, 4:None},
                      'IPSL-CM5A-LR': {0: 1661, 0.5: 1946, 1: 1978, 1.5: 1995, 2: 2014, 2.5: 2033, 3: 2053, 3.5: None, 4:None},
                      'GFDL-ESM2M':   {0: 1661, 0.5: 1974, 1: 2001, 1.5: 2041, 2: 2061, 2.5: None, 3: None, 3.5: None, 4:None},
                      'MIROC5':       {0: 1661, 0.5: 1981, 1: 2008, 1.5: 2037, 2: 2056, 2.5: None, 3: None, 3.5: None, 4:None},},
            'rcp85': {'HadGEM2-ES':   {0: 1661, 0.5: 1985, 1: 1997, 1.5: 2010, 2: 2022, 2.5: 2032, 3: 2041, 3.5: 2050, 4:2057},
                      'IPSL-CM5A-LR': {0: 1661, 0.5: 1946, 1: 1978, 1.5: 1994, 2: 2009, 2.5: 2021, 3: 2031, 3.5: 2040, 4:2049},
                      'GFDL-ESM2M':   {0: 1661, 0.5: 1974, 1: 1999, 1.5: 2021, 2: 2038, 2.5: 2053, 3: 2067, 3.5: None, 4:None},
                      'MIROC5':       {0: 1661, 0.5: 1980, 1: 1999, 1.5: 2018, 2: 2033, 2.5: 2045, 3: 2056, 3.5: 2068, 4:None},},
            }
    }

dict_nyears = {'pre-industrial': 1860-1661+1, 
                   'historical': 2005-1861+1, 
                       'future': 2099-2006+1, 
              'future_extended': 2299-2100+1,
              }

dict_gim = {
    # water sector:
    'waterscarcity': ['H08',],
    'FM': ['H08',],
    }

# Ensemble members...
gcms = [
    'HadGEM2-ES',
#    'IPSL-CM5A-LR',
    'GFDL-ESM2M',
#    'MIROC5',
    ]
ngcm = len(gcms)

dict_tas_coverage = {
    'HadGEM2-ES':   {'piControl': (1661,2299), 'historical': (1861,2005), 'rcp26': (2006,2299), 'rcp60': (2006,2099)},
    'IPSL-CM5A-LR': {'piControl': (1661,2299), 'historical': (1861,2005), 'rcp26': (2006,2299), 'rcp60': (2006,2099)},
    'GFDL-ESM2M':   {'piControl': (1661,2099), 'historical': (1861,2005), 'rcp26': (2006,2099), 'rcp60': (2006,2099)},
    'MIROC5':       {'piControl': (1661,2299), 'historical': (1861,2005), 'rcp26': (2006,2299), 'rcp60': (2006,2099)},
    }

dict_impactindex = {# ncvarname, nckeyname, topic, index_unit, rule(masked_out), threholds
    'waterscarcity': ['water scarcity', 'wsi', 'wsi', 'water', '-', 'less', [0, 0.4]],
    'FM': ['water shortage', 'FM', 'FM', 'water', 'm3/cap/yr', 'greater', [1700]],
    }

dict_item_title = {
    'index': 'indicator ',
    'pop': 'population exposed to',
    'area': 'land area affected by',
    }

dict_unit = {
    'absolute': '% of',
    'absolute-changes': '% of',
    'relative-changes': '%',
    }


countries = [countryname.strip()[:3] for countryname in open(COUNTRY_List_PATH).readlines()]
dict_countryname = {countryname.strip()[:3]: countryname.strip()[4:] for countryname in open(COUNTRY_List_PATH).readlines()}
dict_countrymask = {country: np.ma.make_mask(Dataset(COUNTRYMAP_PATH)['m_{}'.format(country)][:] != 1) for country in countries}


# ----------------------------------------------------------------------------------------------------------------------
# === functions for main processes ===

def read_netcdf(srcpath, ncvarname, period=None):

    #print(srcpath)
    if os.path.isfile(srcpath):
        src = Dataset(srcpath, 'r')[ncvarname][:]
        #print('read_isimip: {} {} {} {}'.format(srcpath, src.shape, src.min(), masked_equal(src, 1e+20).max()))
        src = src.filled(1.e+20)
    else:
        print('WARNING!! {} is not found... Is this OK??'.format(srcpath))
        src = np.full((dict_nyears[period], 360, 720), 1.e+20)
    #print(srcpath, src.shape, src.min(), src.max())
    return src    # <numpy.ndarray>


def read_metricinput(varname, period, scenario):
    """
    read population (and gdp) data
    """
    # TODO: Need to be 2005 pop for all cases?????
    if   period == 'pre-industrial':  soc = '1860soc'; coverage = '1661-1860'
    elif period == 'historical':      soc = 'histsoc'; coverage = '1861-2005'
    elif period == 'future':          soc = '2005soc'; coverage = '2006-2099'
    elif period == 'future_extended': soc = '2005soc'; coverage = '2100-2299'

    if varname == 'pop':
        for_filename = 'population'
        ncvarname = 'number_of_people'
        filename = '{var}_{soc}_0p5deg_annual_{coverage}.nc4'.format(var=for_filename, soc=soc, coverage=coverage)
        srcpath = genpath(METRIC_INPUT_DIR, for_filename, soc, filename)
        src = read_netcdf(srcpath, ncvarname)  # filled with 1e+20
    else: print('check varname...'); sys.exit()

    src = masked_equal(src, 1e+20)
    print('{} {}'.format(srcpath, src.shape))
    return src


def read_ensembledata(ncvarname, nckeyname, period, scenario, gims):  # (ngcm, ngim, nyear, ny, nx)  <masked array>

    def read_impact_index(variable, nckeyname, gcm, gim, period, scenario):#  (nyear, ny, nx)

        data_directory = genpath(DATA_DIR_MAIN, variable, gim.lower(), gcm.lower(), period)

        # TODO: Need to be 2005 pop for all cases?????
        #if period == 'pre-industrial': soc = '1860soc'; coverage = '1661-1860'
        #elif period == 'historical':   soc = 'histsoc'; coverage = '1861-2005' 
        if   period == 'pre-industrial':  soc = '2005soc'; coverage = '1661_1860'  # TODO: this is just tentative....
        elif period == 'historical':      soc = '2005soc'; coverage = '1861_2005'  # TODO: this is just tentative....
        elif period == 'future':          soc = '2005soc'; coverage = '2006_2099'
        elif period == 'future_extended': soc = '2005soc'; coverage = '2100_2299'
        # read input data
        filename = '{gim}_{gcm}_ewembi_{scenario}_{soc}_co2_{index}_global_yearly_{coverage}.nc4'.format(gim=gim.lower(), gcm=gcm.lower(), 
                                                                                                         scenario=scenario.lower(), soc=soc, coverage=coverage, 
                                                                                                         index=variable)
        srcpath = genpath(data_directory, filename)
        src = read_netcdf(srcpath, nckeyname, period)  # (nyear, ny, nx) filled with 1e+20
        if len(src.shape) > 3: # temporal. Just to check data shape...
            print('check the dimension of input data... {}'.format(src.shape)); sys.exit()
        # screening invalid values...
        if src.min() == -np.inf: src[where(isinf(-src))] = 1.e+20; print('-inf was replaced with 1.e+20')
        if src.max() == np.inf:  src[where(isinf(src))] =  1.e+20; print('inf was replaced with 1.e+20')
        if isnan(src.max()):     src[where(isnan(src))] =  1.e+20; print('nan was replaced with 1.e+20')
        print('read_isimip: {} {} {} {}'.format(srcpath, src.shape, src.min(), masked_equal(src, 1e+20).max()))
        return src  # (nyear, ny, nx) <ndarray> filled with 1e+20

    return masked_equal(array([[read_impact_index(ncvarname, nckeyname, gcm, gim, period, scenario) for gim in gims] for gcm in gcms]), 1e+20)  # <ndarray>


def get_indexmask(src, rule, threshold):
    if   rule == 'greater':       return masked_greater(      src, threshold).mask
    elif rule == 'greater_equal': return masked_greater_equal(src, threshold).mask
    elif rule == 'less_equal':    return masked_less_equal(   src, threshold).mask
    elif rule == 'less':          return masked_less(         src, threshold).mask
    else: print('chechk the rule for get_indexmask...'); sys.exit()


def prepare_assessment_results(indexmask, srcindex, pop, area):

    def maskout_by_indexmasks(input_array, indexmask):
        if len(input_array.shape) != len(indexmask.shape): 
            return masked_array(resize(input_array, indexmask.shape), mask=indexmask)  # for area
        else: 
            return masked_array(input_array, mask=indexmask)

    dict_results = {
        'index': maskout_by_indexmasks(srcindex, indexmask),  # (ngcm, ngim, nyear, ny, nx)
        'pop':   maskout_by_indexmasks(pop,      indexmask),  # (ngcm, ngim, nyear, ny, nx)
        'area':  maskout_by_indexmasks(area,     indexmask),  # (ngcm, ngim, nyear, ny, nx)
        }
    return dict_results


def gen_countrysrc(item, src_type, globalsrc, countrymask):
    if not (item == 'index' and src_type == 'reference'):
        return masked_array(globalsrc, mask=resize(countrymask, globalsrc.shape))
    else:
        return None


def calc_country_value(item, countrysrc, countrysrc_metric=None):   # countrysrc.shape =
    """
    Spacial summary of country source...
    :param item: index, pop, area
    :param countrysrc: (ngcm, ngim, nyear, ny, nx) MaskedArray
    :param countrysrc_metric: index has Nane, pop has a MaskedArray (nyear, ny, nx) , area has a MaskedArray (ny, nx)
    :return: an time series Array  (ngcm, ngim, nyear)
    """
    if item == 'index':
        country_value = countrysrc.mean(axis=(3, 4))  #  TODO: Make sure if this is correct.
    elif item == 'pop':
        country_value = np.divide(countrysrc.sum(axis=(3, 4)), countrysrc_metric.sum(axis=(1,2))) * 100 #  [%]
    elif item == 'area':
        country_value = np.divide(countrysrc.sum(axis=(3, 4)), countrysrc_metric.sum()) * 100 #  [%]
    return country_value  # (ngcm, ngim, nyear)


def concatenator(srcs, syear, eyear, ngcm, ngim):
    print('in concatenator, syear={}, eyear={}'.format(syear, eyear))
    src_preindustrical, src_historical, src_future, src_future_extended = srcs
    if src_preindustrical is None: src_preindustrical = np.full((ngcm, ngim, dict_nyears['pre-industrial']), 1.e+20)
    if src_historical is None: src_historical = np.full((ngcm, ngim, dict_nyears['historical']), 1.e+20)
    if src_future is None: src_future = np.full((ngcm, ngim, dict_nyears['future']), 1.e+20)
    if src_future_extended is None: src_future_extended = np.full((ngcm, ngim, dict_nyears['future_extended']), 1.e+20)
    src = np.concatenate([src_preindustrical, src_historical, src_future, src_future_extended], axis=2)  # (ngcm, ngim, nyear_all)
    if 1661 < syear: src[:,:,:years_full.index(syear)] = 1.e+20
    if eyear < 2299: src[:,:,years_full.index(eyear)+1:] = 1.e+20
    src = masked_equal(src, 1.e+20)
    return src


def calc_memberstats_for_this_period(src, temporal_type, scenario, prd, output_type, gims):
    """
    :param src: (ngcm, ngim, nrcp, nyear_long).  nrcp is 1 (for timeslices), or 2 (for temperature-change)
    :param temporal_type: timeslices or temperature-change
    :param rcp: piControl, historical, rcp26 or rcp60 (for timeslices), or mix (for temperature-change)
    :param prd: the timing to caclulate about
    :param output_type: absolute, absolute-changes, relative-changes
    :return: representative values of a preiod     (ngcm, ngim), (ngcm, ngim), (ngcm, ngim), (ngcm, ngim),
    """

    def gen_index(temporal_type, _scenario, gcm, prd):
        # get index to cut out data (this is gcm dipendant variable)
        if temporal_type == 'timeslices':
            index_start = years_full.index(dict_timing[temporal_type][prd][0])
            index_end   = years_full.index(dict_timing[temporal_type][prd][1])
        elif temporal_type == 'temperature-change':
            syear = dict_timing[temporal_type][_scenario][gcm][prd]
            index_start = years_full.index(syear) if not syear is None else None
            index_end   = index_start + 30 if not syear is None else None   # TODO: 31 years mean???
        return index_start, index_end

    def extract_samples_from_a_scenario(_src, prd, i_scenario, _scenario, gcm, temporal_type):
        index_start, index_end = gen_index(temporal_type, _scenario, gcm, prd)
        if index_start is None: return np.array([None])
        else: return _src[i_scenario, index_start:index_end+1]
   
    def calc_stats(_src, stat_type, prd, scenarios, gcm, temporal_type, output_type, basis=None):

        nexted_samples = [extract_samples_from_a_scenario(_src, prd, i_scenario, _scenario, gcm, temporal_type).tolist() 
                                                                        for i_scenario, _scenario in enumerate(scenarios)]
        samples = [value for inner_samples in nexted_samples for value in inner_samples]
        samples = masked_equal(samples, 1e+20)
        samples = masked_equal(samples, None)
        samples = samples.compressed()

        if samples.size is not 0:
            if  output_type == 'absolute': pass
            elif output_type == 'absolute-changes': samples = samples - basis
            elif output_type == 'relative-changes': samples = np.divide(samples-basis, basis) if not basis == 0 else [0]
            if stat_type == 'mean': value = np.mean(samples)
            elif stat_type == 'std': value = np.std(samples)
            print('sample: {} >>> value: {}'.format(samples, value))
            return value
        else: 
            return None

    # preparations 
    mean_of_this_period         = np.full((len(gcms), len(gims)), None)  # (ngcm, ngim)
    std_of_this_period          = np.full((len(gcms), len(gims)), None)  # (ngcm, ngim)
    lower_border_of_this_period = np.full((len(gcms), len(gims)), None)  # (ngcm, ngim)
    upper_border_of_this_period = np.full((len(gcms), len(gims)), None)  # (ngcm, ngim)

    if temporal_type == 'timeslices': scenarios = [scenario]
    elif temporal_type == 'temperature-change': scenarios = scenarios_rcp


    # calculate member-specific stats from sample (nrcp, nyears_for_the_period)
    for igcm, gcm in enumerate(gcms):
        for igim, gim in enumerate(gims):

            if output_type == 'absolute':
                mean_of_this_period[igcm, igim] = calc_stats(src[igcm, igim], 'mean', prd, scenarios, gcm, temporal_type, output_type)
                std_of_this_period[igcm, igim] = calc_stats(src[igcm, igim], 'std', prd, scenarios, gcm, temporal_type, output_type)
            elif 'changes' in output_type:
                basis_endindex = years_full.index(1860)
                basis = src[igcm, igim, :, :basis_endindex+1].mean()  # TODO: is base-peiod pre-industrial? or piCont in the same period??
                mean_of_this_period[igcm, igim] = calc_stats(src[igcm, igim], 'mean', prd, scenarios, gcm, temporal_type, output_type, basis)
                std_of_this_period[igcm, igim] = calc_stats(src[igcm, igim], 'std', prd, scenarios, gcm, temporal_type, output_type, basis)

            if not mean_of_this_period[igcm, igim] is None:
                lower_border_of_this_period[igcm, igim] = mean_of_this_period[igcm, igim] - std_of_this_period[igcm, igim]
                upper_border_of_this_period[igcm, igim] = mean_of_this_period[igcm, igim] + std_of_this_period[igcm, igim]
            else:
                lower_border_of_this_period[igcm, igim] = None
                upper_border_of_this_period[igcm, igim] = None
    
    return (mean_of_this_period.astype('float64'), 
            std_of_this_period.astype('float64'), 
            lower_border_of_this_period.astype('float64'), 
            upper_border_of_this_period.astype('float64'))  # (ngcm, ngim)


def calc_median(src_input):
    if src_input.shape == (0,): return None
    else: return np.median(src_input)


def gen_border_value(median_value, std_value, border_type):
    """
    :param median_value:
    :param std_value: 
    :param border_type: shading_lower_border, shading_upper_border 
    """
    if not median_value is None:
        if border_type == 'shading_lower_border': return median_value - std_value
        elif border_type == 'shading_upper_border': return median_value + std_value
        else: print('check border_type'); sys.exit()
    else: return None


# === functions for output ===
def create_dict_for_json_slice(dict_countryresult, **kwargs):
    """
    For case.1  (see examples at the end of this code...)
    :param dict_countryresult:
    :param kwargs:
    :return:
    """

    temporal_type = kwargs['temporal_type']  # years, tempereture-changes
    output_type   = kwargs['output_type']    # absolute-values, absolute-changes, relative-changes
    impact_index  = kwargs['impact_index']   # drought, FM
    index_name    = kwargs['index_name']     # drought, water shortage
    item          = kwargs['item']           # index, area, pop
    itemunit      = kwargs['itemunit']
    threshold     = kwargs['threshold']
    gcms          = kwargs['gcms']
    gims          = kwargs['gims']
    country       = kwargs['country']

    print('output_type: {}'.format(output_type))

    # variable name
    if output_type == 'absolute-changes' or output_type == 'relative-changes':
        # ex, 'absolute change in land area affected by drought'
        variable = '{} in {} {}'.format(output_type.replace('-', ' '), dict_item_title[item], index_name) 
    elif output_type == 'absolute':
        # ex, 'land area affected by drought'
        variable = '{} {}'.format(dict_item_title[item], index_name)
    else: print('check output_type...'); sys.exit()

    # x_axis_list
    if temporal_type == 'timeslices': x_axis_list = 'timeslice_list'
    elif temporal_type == 'temperature-change': x_axis_list = 'temperature_list'

    # plot_title
    if temporal_type == 'timeslices': plot_title = '{} vs. {}'.format(variable.capitalize(), 'Time')
    elif temporal_type == 'temperature-change': plot_title = '{} vs. {}'.format(variable.capitalize(), 'Global warming level')

    # plot_type
    if temporal_type == 'timeslices': plot_type = 'indicator_vs_time'
    elif temporal_type == 'temperature-change': plot_type = 'indicator_vs_temperature'

    # plot_label_x
    if temporal_type == 'timeslices': plot_label_x = 'Year'
    elif temporal_type == 'temperature-change': plot_label_x = 'Global warming level'

    # plot_unit_x
    if temporal_type == 'timeslices': plot_unit_x = ''
    elif temporal_type == 'temperature-change': plot_unit_x = '\u00b0C'

    # plot_unit_y
    if 'absolute' in output_type: plot_unit_y = dict_unit[output_type]+dict_item_title[item][:10]
    else: plot_unit_y = dict_unit[output_type]

    # attribute
    dict_attr = {'author': WHO_ARE_YOU,
                 'data': datetime.datetime.today().strftime('%Y-%m-%d'),
                 'comment': {'threshold': threshold,
                             'code version': code_version,
                             'comment': 'This is jus a test file. Needs to be updated...'
                             },
                 }

    dict_out = {'assessment_category': assessment_category,
                'indicator': index_name,
                'variable': variable,
                'region': dict_countryname[country],
                'climate_model_list': gcms,
                'impact_model_list': gims,
                x_axis_list: dict_prds[temporal_type],
                'data': dict_countryresult,
                'plot_title': plot_title,
                'plot_type': plot_type,
                'plot_label_x': plot_label_x,
                'plot_label_y': variable.capitalize(),
                'plot_unit_x': plot_unit_x,
                'plot_unit_y': plot_unit_y,
                'esgf_search_url': "??????",
                'file_attribute': dict_attr,
                }

    # only for versus-timeslices
    if temporal_type == 'timeslices':
        dict_tmp = {
            'climate_scenario_list': scenarios_all,
            'n_timeslices': len(dict_prds[temporal_type]),
            }
        dict_out = {**dict_out, **dict_tmp}

    return dict_out


def read_temperature_info(scenario, gcm):
    syear, eyear = dict_tas_coverage[gcm][scenario]
    filename = 'tas_day_{}_{}_r1i1p1_EWEMBI_{}-{}.fldmean.yearmean.txt'.format(gcm, scenario, syear, eyear)
    srcpath = os.path.join(STEFAN_FILE_DIR, gcm, filename)
    print('read tas file: {}'.format(srcpath))
    lines = open(srcpath).readlines()[1:]
    return np.array([float(re.split(' +', line.strip())[1]) for line in lines])


def gen_temperature_dictionary():
    dict_tas = {scenario: 
                    {gcm: 
                        {'global_mean_temperatures': read_temperature_info(scenario, gcm)}
                    for gcm in gcms } 
                for period, scenario in periods_scenarios}
    for scenario, gcm in product(scenarios_all, gcms):
        # TODO: this is period right???  1661-1860???
        dict_tas[scenario][gcm]['piControl_avg_temperature '] = dict_tas['piControl'][gcm]['global_mean_temperatures'][:years_full.index(1861)].mean()  
        dict_tas[scenario][gcm]['global_warming'] = dict_tas[scenario][gcm]['global_mean_temperatures'] - dict_tas[scenario][gcm]['piControl_avg_temperature ']
    for scenario, gcm, tas_item in product(scenarios_all, gcms, ['global_mean_temperatures', 'global_warming']):
        syear, eyear = dict_tas_coverage[gcm][scenario]
        header = [None] * (syear - 1661) if 1661 < syear else []
        footer = [None] * (2299 - eyear) if eyear < 2299 else []
        dict_tas[scenario][gcm][tas_item] = header + dict_tas[scenario][gcm][tas_item].tolist() + footer
    return dict_tas


def create_dict_for_json_yearly(dict_countrysrc, **kwargs):  # dict_countrysrc = {scenario: (ngcm, ngim, nyear)}

    country       = kwargs['country']
    impact_index  = kwargs['impact_index']   # drought, FM
    index_name    = kwargs['index_name']     # drought, water shortage
    item          = kwargs['item']           # index, area, pop
    itemunit      = kwargs['itemunit']
    threshold     = kwargs['threshold']
    temporal_type = kwargs['temporal_type']  # years, tempereture-changes
    output_type   = kwargs['output_type']    # absolute-values, absolute-changes, relative-changes
    gcms          = kwargs['gcms']
    gims          = kwargs['gims']

    if output_type == 'absolute-changes' or output_type == 'relative-changes':
        # ex, 'absolute change in land area affected by drought'
        variable = '{} in {} {}'.format(output_type.replace('-', ' '), dict_item_title[item], index_name)  
    elif output_type == 'absolute':
        # ex, 'land area affected by drought'
        variable = '{} {}'.format(dict_item_title[item], index_name)  
    else: print('check output_type...'); sys.exit()

    if 'absolute' in output_type: plot_unit_y = dict_unit[output_type]+dict_item_title[item][:10]
    else: plot_unit_y = dict_unit[output_type]

    dict_attr = {'author': WHO_ARE_YOU,
                 'data': datetime.datetime.today().strftime('%Y-%m-%d'),
                 'comment': {'threshold': threshold,
                             'code version': code_version,
                             'comment': 'This is jus a test file. Needs to be updated...'
                             },
                 }

    dict_temperature = gen_temperature_dictionary()

    dict_result = {scenario: 
                      {gcm: 
                         {**dict_temperature[scenario][gcm], **{'runs': {}}}
                      for gcm in gcms} 
                   for period, scenario in periods_scenarios}
    # for runs...
    for period, scenario in periods_scenarios:
        for igcm, gcm in enumerate(gcms):
            syear, eyear = dict_tas_coverage[gcm][scenario]
            header = [None] * (syear - 1661) if 1661 < syear else []
            footer = [None] * (2299 - eyear) if eyear < 2299 else []
            for igim, gim in enumerate(gims):
                dict_result[scenario][gcm]['runs'][gim] = header + dict_countrysrc[(period, scenario)][igcm, igim].tolist() + footer

    dict_out = {'assessment_category': assessment_category,
                'indicator': index_name,
                'variable': variable,
                'region': dict_countryname[country],
                'year_list': list(years_full),
                'climate_scenario_list': scenarios_all,
                'climate_model_list': gcms,
                'impact_model_list': gims,
                'data': dict_result,
                'plot_title': '{} vs. Time'.format(variable.capitalize()),
                'plot_type': 'indicator_vs_timeline',
                'plot_label_x': 'Year',
                'plot_label_y': variable.capitalize(),
                'plot_unit_x': '',
                'plot_unit_y': plot_unit_y,
                'esgf_search_url': "??????",
                'file_attribute': dict_attr,
                }

    return dict_out


def writeout_json(dictionary, jsonfilename, outdir):
    if JSONOUTLOUD: print('\n{}  = {}'.format(jsonfilename, dictionary))
    outpath = genpath(outdir, jsonfilename)
    with open(outpath, 'w') as jsonfile:
        json.dump(dictionary, jsonfile, indent=4, separators=(',', ': '), ensure_ascii=False, sort_keys=True, ignore_nan=True)
        print('savejson: {}'.format(outpath))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main(*args):
    print(args)
    start = time.time()

    print('reading metric data...')
    dict_pop = {(period, scenario): read_metricinput('pop', period, scenario) for period, scenario in periods_scenarios}     # {scenario: (nyear, ny, nx)}
    area = np.fromfile(AREAPATH, 'float32').reshape(360, 720) * 1e-9                                                         #                    (ny, nx)   [km2]

    for impact_index in impact_indices:

        index_name = dict_impactindex[impact_index][0]
        ncvarname  = dict_impactindex[impact_index][1]
        nckeyname  = dict_impactindex[impact_index][2]
        topic      = dict_impactindex[impact_index][3]
        index_unit = dict_impactindex[impact_index][4]
        rule       = dict_impactindex[impact_index][5]
        thresholds = dict_impactindex[impact_index][6]

        # depending on indicator, its unit and gims available for the index differ...
        dict_mtr_unit = {'index': index_unit, 'pop': 'cap', 'area': 'km2'}
        gims = dict_gim[impact_index]
        ngim = len(gims)

        # read index and generate dict_srcindex   {scenarios: (ngcm, ngim, nyear, ny, nx)}
        print('\nread index data...')
        dict_srcindex = {(period, scenario): read_ensembledata(ncvarname, nckeyname, period, scenario, gims) for period, scenario in periods_scenarios}
        print('\ndict_srcindex = {')
        for period, scenario in periods_scenarios:
            print('   {}: {} {}'.format((period, scenario), dict_srcindex[(period, scenario)].shape, type(dict_srcindex[(period, scenario)])))
        print('   }')

        # generate indexmasks for each threshold   {scenarios: {thresholds: (ngcm, ngim, nyear, ny, nx)}}
        dict_indexmasks = {(period, scenario):
                               {threshold:
                                     get_indexmask(dict_srcindex[(period, scenario)], rule, threshold)  # {scenario: threshold: (ngcm, ngim, nyear, ny, nx)}
                                for threshold in thresholds}
                           for period, scenario in periods_scenarios}
        print('\nif threshold is {}'.format(thresholds[0]))
        print('dict_indexmaskes = {')
        for period, scenario in periods_scenarios:
            print('   {}: {}'.format((period, scenario), dict_indexmasks[(period, scenario)][thresholds[0]].shape))
        print('   }')

        # generate affected information (index and metrics) for each scenarios and threshold   (global maps)
        # {scenarios: {threholds: {metrixes:  (ngcm, ngim, nyear, ny, nx)}}}
        dict_assessment = {(period, scenario):
                               {threshold:
                                    prepare_assessment_results(dict_indexmasks[(period, scenario)][threshold], 
                                                               dict_srcindex[(period, scenario)],
                                                               dict_pop[(period, scenario)],
                                                               area)
                                for threshold in thresholds}
                           for period, scenario in periods_scenarios}

        dict_metric_dat = {(period, scenario): {'index': None, 'pop': dict_pop[(period, scenario)], 'area': area} for period, scenario in periods_scenarios}


        # <main loop>
        for item, threshold in product(items, thresholds):  # items = ['index', 'pop', 'area']
            print('\n\n--- processing for {} {}'.format(item, threshold))

            outputdir = genpath(OUTPUT_DIR, index_name.replace(' ','-'), assessment_category)
            if not os.path.isdir(outputdir): os.makedirs(outputdir)

            # start country assessment
            for country in tqdm(countries):
                print('\n\n{}'.format(country))

                outputdir_country = genpath(outputdir, country)
                if not os.path.isdir(outputdir_country): os.makedirs(outputdir_country)

                # link and get country mask
                countrymask = dict_countrymask[country]

                # dict_countrysrc = {scenario: (ngcm, ngim, nyear)}  
                # every scenario has different nyear here
                # TODO: check calc_country_value. mean? sum?
                dict_countrysrc = {(period, scenario): 
                                        calc_country_value(
                                            item, 
                                            gen_countrysrc(item, 'assessment', dict_assessment[(period, scenario)][threshold][item], countrymask), 
                                            gen_countrysrc(item, 'reference',  dict_metric_dat[(period, scenario)][item],            countrymask),) 
                                                                                                             for period, scenario in periods_scenarios}

                print('\n==================\ncreate country data set\n==================\n')
                # <inter-main loop>
                # output results for each period
                for temporal_type in temporal_types:
                    print('\n======\n{}\n======'.format(temporal_type))
                    # temporal_types are [years, temperature-change]

                    # concatenate pre-industrial, historical and a rcp    
                    # TODO: Is this process for historical preiod OK????? mean of two rcps??
                    if temporal_type == 'timeslices': 
                        # all scenarios with proper maskout
                        dict_countrysrc_rcp = {
                            'piControl': concatenator([dict_countrysrc[periods_scenarios[0]], None, None, None], 1661, 1860, ngcm, ngim),
                            'historical': concatenator([None, dict_countrysrc[periods_scenarios[1]], None, None], 1861, 2000, ngcm, ngim),
                            'rcp26': concatenator([None, dict_countrysrc[periods_scenarios[1]], 
                                                   dict_countrysrc[('future', 'rcp26')], dict_countrysrc[('future_extended', 'rcp26')]],
                                                   2001, 2299, ngcm, ngim),
                            'rcp60': concatenator([None, dict_countrysrc[periods_scenarios[1]], 
                                                   dict_countrysrc[('future', 'rcp60')], dict_countrysrc[('future_extended', 'rcp60')]], 
                                                   2001, 2299, ngcm, ngim),
                            }
                    elif temporal_type == 'temperature-change':  
                        # only two rcps for full period without any maskout
                        dict_countrysrc_rcp = {rcp: concatenator([dict_countrysrc[periods_scenarios[0]], dict_countrysrc[periods_scenarios[1]],
                                                                  dict_countrysrc[('future', rcp)], dict_countrysrc[('future_extended', rcp)]], 
                                                                  1661, 2299, ngcm, ngim, )
                                                    for rcp in scenarios_rcp}

                    if temporal_type == 'timeslices': scenarios_for_the_temporal_type = scenarios_all  # piControl, historical, rcp26, rcp60
                    elif temporal_type == 'temperature-change': scenarios_for_the_temporal_type = ['mix']  # scenario mix

                    for output_type in output_types:  # output_types are ['absolute', 'absolute-changes', 'relative-changes'], 

                        # Initialize a brank dictionary for countries, to archive time series data for JSON files. 
                        # The deepest key will be stats (median, mean, std)
                        dict_various_countryvalues = {scenario:  # [piControl, historical, rcp26, rcp60] or [mix]
                                                         {gcm:  # each gcm  or  overall
                                                             {ensemble_stats_type:  # median, std, lower, upper, runs
                                                                 {gim:  # gim under "runs"
                                                                     {run_stats_type:  # mean, std, lower, upper
                                                                           []  # empty list for individual results
                                                                      for run_stats_type in run_stats_types}
                                                                  for gim in gims} if (gcm != 'overall' and ensemble_stats_type == 'runs') else []
                                                              for ensemble_stats_type in ensemble_stats_types}
                                                          for gcm in ['overall']+gcms}
                                                      for scenario in scenarios_for_the_temporal_type}
                        for scenario in scenarios_for_the_temporal_type:
                            del dict_various_countryvalues[scenario]['overall']['runs']  # Don't need ['overall']['runs']
                        #print(json.dumps(dict_various_countryvalues, indent=4, separators=(',', ': '), ensure_ascii=False, sort_keys=True))


                        for scenario in scenarios_for_the_temporal_type:  # scenarios is ['piControl', 'historical', 'rcp26', 'rcp60'] or ['mix']

                            # link and pick-up a target country's src
                            if scenario == 'mix': # e.q. temporal_type == 'temperature-change'
                                country_values_fullperiod = np.transpose(np.stack([dict_countrysrc_rcp[rcp] for rcp in scenarios_rcp]), axes=(1,2,0,3)) 
                                # (ngcn, ngim, 2, nyear_long) <ndarray>
                                # TODO: this covers two rcps, which has different periods for a target global warming level
                            else:
                                country_values_fullperiod = np.transpose(np.array([dict_countrysrc_rcp[scenario]]), axes=(1,2,0,3))
                                # (ngcn, ngim, 1, nyear_long) <ndarray>
                            country_values_fullperiod = masked_equal(country_values_fullperiod, 1.e+20)  # <MaskedArray>

                            # link and pickup a target empty dictionary
                            targetdict = dict_various_countryvalues[scenario]

                            # generate periods' values...
                            # for each ensemble member during each period
                            for iprd, prd in enumerate(dict_prds[temporal_type]):
                                # 'timeslices': [(1661, 1680), (1681, 1700), ..., (2241, 2260), (2261, 2280)]
                                # 'temperature-change': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

                                # member_values: memberspecific mean, std, lower_border, upper_border  (4 variables) 
                                # each array shape is (ngcm, ngim, nrcp) <class 'numpy.ma.core.MaskedArray'>
                                member_values = calc_memberstats_for_this_period(country_values_fullperiod, 
                                                                                 temporal_type, scenario, prd, 
                                                                                 output_type, gims)
                                print('\nmember_values: {}'.format(member_values))

                                # 1st: each member
                                for run_stats_type, member_value in zip(run_stats_types, member_values):
                                    for igcm, gcm in enumerate(gcms):
                                        for igim, gim in enumerate(gims):
                                            targetdict[gcm]['runs'][gim][run_stats_type].append(member_value[igcm, igim])

                                # 2nd: ensemble for a GCM  (ensemble_stats_types: median,std,lower,upper)
                                for ensemble_stats_type, member_value in zip(ensemble_stats_types[:2], member_values[:2]):
                                    for igcm, gcm in enumerate(gcms):
                                        country_value = calc_median(masked_equal(member_value[igcm], None).compressed())  # single value
                                        targetdict[gcm][ensemble_stats_type].append(country_value)
                                for ensemble_stats_type in ensemble_stats_types[2:4]:
                                    for igcm, gcm in enumerate(gcms):
                                        country_value = gen_border_value(targetdict[gcm]['median'][-1], 
                                                                         targetdict[gcm]['interannual_standard_deviation'][-1], 
                                                                         ensemble_stats_type)
                                        targetdict[gcm][ensemble_stats_type].append(country_value)

                                for ensemble_stats_type, member_value in zip(ensemble_stats_types[:2], member_values[:2]):
                                    country_value = calc_median(masked_equal(member_value, None).compressed())  # single value
                                    targetdict['overall'][ensemble_stats_type].append(country_value)
                                for ensemble_stats_type in ensemble_stats_types[2:4]:
                                    country_value = gen_border_value(targetdict['overall']['median'][-1], 
                                                                     targetdict['overall']['interannual_standard_deviation'][-1], 
                                                                     ensemble_stats_type)
                                    targetdict['overall'][ensemble_stats_type].append(country_value)


                        print('\n==================\nwrite out JSON files\n==================\n')
                        if temporal_type == 'timeslices': targetdict = dict_various_countryvalues
                        elif temporal_type == 'temperature-change': targetdict = dict_various_countryvalues['mix']

                        # out JSON: versus-timeslices or versus-temperature-change
                        dict_out = create_dict_for_json_slice(targetdict, country=country, 
                                                              impact_index=impact_index, index_name=index_name, item=item, itemunit=dict_mtr_unit[item],
                                                              threshold=threshold, temporal_type=temporal_type, output_type=output_type,
                                                              gcms=gcms, gims=gims)
                        json_filename = '{}-{}{}_{}_versus-{}_{}.json'.format(dict_item_title[item].replace(' ', '-'), index_name.replace(' ', '-'),
                                                                              '-{}'.format(output_type) if not output_type == 'absolute' else '',
                                                                              assessment_category, temporal_type, country)
                        country_json_outdir = outputdir_country
                        writeout_json(dict_out, json_filename, country_json_outdir)

                        if temporal_type == 'timeslices':
                            # out JSON: versus-years
                            dict_out = create_dict_for_json_yearly(dict_countrysrc, country=country, 
                                                                   impact_index=impact_index, index_name=index_name, 
                                                                   item=item, itemunit=dict_mtr_unit[item],
                                                                   threshold=threshold, temporal_type=temporal_type, output_type=output_type,
                                                                   gcms=gcms, gims=gims)
                            json_filename = '{}-{}{}_{}_versus-years_{}.json'.format(dict_item_title[item].replace(' ', '-'), 
                                                                                     index_name.replace(' ', '-'),
                                                                                     '-{}'.format(output_type) if not output_type == 'absolute' else '',
                                                                                     assessment_category, country)
                            country_json_outdir = outputdir_country
                            writeout_json(dict_out, json_filename, country_json_outdir)



                print('\n')

    process_time = time.time() - start
    print(process_time)


if __name__=='__main__':
    main(*sys.argv)


"""

===================================
Dimensions of ISIpedia items
===================================

data_cube (top directory)
    |
    +--- drought (2nd directory)
              |
              +--- ISIMIP-projections (3rd directory)
                      |
                      + --- MAR (4th directory)
                             |
                             +------ land-area
                             |          |
                             |          +--- versus-years
                             |          |           |
                             |          |           +--- land-area-affected-by-drought_ISIMIP-projections_versus-years_MARs.json
                             |          |           |
                             |          |           +--- land-area-affected-by-drought-relative-changes_ISIMIP-projections_versus-years_MAR.json
                             |          |           |
                             |          |           +--- land-area-affected-by-drought-absolute-changes_ISIMIP-projections_versus-years_MAR.json
                             |          |
                             |          +--- versus-temperature-change
                             |          |           |
                             |          |           +--- land-area-affected-by-drought_ISIMIP-projections_versus-temperature-change_MAR.json
                             |          |           |
                             |          |           +--- land-area-affected-by-drought-relative-changes_ISIMIP-projections_versus-temperature-change_MAR.json
                             |          |           |
                             |          |           +--- land-area-affected-by-drought-absolute-changes_ISIMIP-projections_versus-temperature-change_MAR.json
                             |          |
                             |          +--- versus-timeslices
                             |                      |
                             |                      +--- land-area-affected-by-drought_ISIMIP-projections_versus-timeslices_MAR.json
                             |                      |
                             |                      +--- land-area-affected-by-drought-relative-changes_ISIMIP-projections_versus-timeslices_MAR.json
                             |                      |
                             |                      +--- land-area-affected-by-drought-absolute-changes_ISIMIP-projections_versus-timeslices_MAR.json
                             |
                             |
                             +------ population
                                        |
                                        +--- versus-years
                                        |           |
                                        |           +--- population-affected-by-drought_ISIMIP-projections_MAR_versus-years.json
                                        |           |
                                        :           :
                                        :


( CASE.1 )

in a JSON file: ex.

--- temperature-change ---

  {"assessment_category": "ISIMIP-projections",
   "indicator": "drought"
   "variable": "absolute changes in land area affected by drought"
   "climate_model_list": ["IPSL-CM5A-LR", "GFDL-ESM2M", "MIROC5"],
   "impact_model_list": ['IM_1', 'IM_2', ...],
   "temperature_list": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]                                     # or year_list
   "data": {
       "GCM_1": {
           "median": [...],
           "interannual_standard_deviation": [...],
           "shading_lower_border": [...],
           "shading_upper_border": [...]
           "runs": {
               "IM_1": {
                  "mean":[...],
                  "interannual_standard_deviation": [...],
                  "shading_lower_border": [...],
                  "shading_upper_border": [...]
                  }
                      :
               },
           },
             :
             :
             :
       "overall": {
           "median": [...],
           "interannual_standard_deviation": [...],
           "shading_lower_border": [...],
           "shading_upper_border": [...]
           }
       },
   "plot_title": "Land area affected by drought (absolute changes) vs. Global warming level",  # or vs. Time
   "plot_type": "indicator_vs_temperature",                                                    # or vs_timeline
   "plot_label_x": "Global warming level",                                                     # or "Year"
   "plot_label_y": "Land area affected by drought (abs. changes)",
   "plot_unit_x": "\u00b0C",                                                                   # ????
   "plot_unit_y": "% of land area",
   "region": "Morocco",
   "esgf_search_url": "??????",
   }


--- years ---

  {"assessment_category": "ISIMIP-projections",
   "indicator": "drought"
   "variable": "absolute changes in land area affected by drought"
   "climate_scenario_list": ["rcp26", "rcp60"],
   "climate_model_list": ["IPSL-CM5A-LR", "GFDL-ESM2M", "MIROC5"],
   "impact_model_list": ["CLM45"],
   "year_list": [2006-2020, 2021-2040, 2041-2060, 2061-2080, 2081-2099]
   "data": {
       "rcp26": {
           "GCM_1": {
               "median": [...],
               "interannual_standard_deviation": [...],
               "shading_lower_border": [...],
               "shading_upper_border": [...]
               "runs": {
                   "IM_1": {
                      "mean":[...],
                      "interannual_standard_deviation": [...],
                      "shading_lower_border": [...],
                      "shading_upper_border": [...]
                      }
               }
           },
               :
               :
       },
       "rcp60": {
           "GCM_1": {
               "global_mean_temperatures": [...],
               "global_warming": [...],
               "piControl_avg_temperature": 286.743,
               "runs": {
                   "CLM45": [...]
               }
           },
               :
               :
       }
   },
   "indicator": "land-area-affected-by-drought",
   "plot_title": "Land area affected by drought (absolute changes) vs. Time",
   "plot_type": "indicator_vs_timeline",
   "plot_label_y": "Land area affected by drought (abs. changes)",
   "plot_label_x": "Year",
   "plot_unit_x": "",
   "plot_unit_y": "% of land area",
   "region": "Morocco",
   "esgf_search_url": "?????",
   }




( CASE.2 )


--- years v.1---  (yearly)

  {"assessment_category": "ISIMIP-projections",
   "indicator": "drought"
   "variable": "absolute changes in land area affected by drought"
   "climate_scenario_list": ["rcp26", "rcp60"],
   "climate_model_list": ["IPSL-CM5A-LR", "GFDL-ESM2M", "MIROC5"],
   "impact_model_list": ["CLM45"],
   "year_list": [1661, 1662, 1663, 1664, ..., 2296, 2297, 2298, 2299],
   "data": {
       "historical": {
           "GCM_1": {
               "global_mean_temperatures": [null, null, null, ..., null, null, 286.62, 286.723, ..., 287.479, 287.572, null, null, ...],
               "global_warming": [null, null, null, ..., null, null, -0.123, -0.02, ..., 0.736, 0.829, null, null, ...],
               "piControl_avg_temperature": 286.743,
               "runs": {
                   "CLM45": [null, null, null, ..., null, null, -0.1690512, -0.1753233, ..., -0.4932155, 18.01531, null, null, ...]
               }
           },
               :
               :
      },
       "piControl": {
           "GCM_1": {
               "global_mean_temperatures": [...],
               "global_warming": [...],
               "piControl_avg_temperature": 286.743,
               "runs": {
                   "CLM45": [...]
               }
           },
               :
               :
       },
       "rcp26": {
               :
               :
       },
       "rcp60": {
               :
               :
       }
   },
   "indicator": "land-area-affected-by-drought",
   "plot_title": "Land area affected by drought (absolute changes) vs. Time",
   "plot_type": "indicator_vs_timeline",
   "plot_label_y": "Land area affected by drought (abs. changes)",
   "plot_label_x": "Year",
   "plot_unit_x": "",
   "plot_unit_y": "% of land area",
   "region": "Morocco",
   "esgf_search_url": "?????",
   }

"""
