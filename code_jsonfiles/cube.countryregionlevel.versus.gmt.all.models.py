#!/usr/bin/python



import numpy as np
import sys
sys.path.append('../')
import counting_extremes as ce
from optparse import OptionParser



usage = 'usage: python %prog [options]'
parser = OptionParser(usage)
parser.add_option('-s', '--sec', action='store', type='string', dest='s', help='impact sector')
parser.add_option('-m', '--mods', action='store', type='string', dest='m', help='impact model names concatenated by underscores')
parser.add_option('-g', '--gcms', action='store', type='string', dest='g', help='GCM names concatenated by underscores')
parser.add_option('-c', '--cor', action='store', type='string', dest='c', help='country code or region name')
parser.add_option('-p', '--pop', action='store_true', dest='p', default=False, help='plot population fraction affected (default: plot area fraction affected)')
parser.add_option('-r', '--rel', action='store_true', dest='r', default=False, help='compute relative change (default: absolute change)')
parser.add_option('-i', '--iav', action='store_true', dest='i', default=False, help='load interannual variability (default: load multi-year mean change)')
parser.add_option('-n', '--noc', action='store_true', dest='n', default=False, help='load no-change data (default: load change data)')
(options, args) = parser.parse_args()
sec = options.s
mods = options.m.split('_')
gcms = options.g.split('_')
cor = options.c
lpopulation = options.p
lrelative = options.r
liav = options.i
lnoc = options.n
lmultiplesectors = sec[:4] in ['sum+', 'max+']



relfix = '.relative' if lrelative else ''
secfix = '.multiplesectors' if lmultiplesectors else '.singlesector'
popfix = 'PFA' if lpopulation else 'AFA'
nocfix = 'nochange_' if lnoc else ''
idir = 'txt/'+popfix+'.countryregionlevel.versus.gmt.and.timeseries'+secfix+relfix



# load multi-year mean changes or interannual variabilities per warming level
mean_changes_per_warming_level = np.empty((ce.warming_levels.size, len(gcms), len(mods)), dtype=float)
for g, gcm in enumerate(gcms):
    for m, mod in enumerate(mods):
        ifile = nocfix+popfix+relfix.replace('.', '_')+'_'+sec+'_'+mod+'_'+gcm+'_'+cor
        mean_changes_per_warming_level[:,g,m] = np.loadtxt(idir+'/'+ifile+'.txt', skiprows=2-lnoc, usecols=((1 if lnoc else 14)+liav,))

# identify data gaps in the gcm x mod matrix
gaps_gcms_mods = np.all(np.isnan(mean_changes_per_warming_level), axis=0)  # shape (len(gcms), len(mods))
gaps_mods = np.any(gaps_gcms_mods, axis=0)
gapless_mods = np.logical_not(gaps_mods)

if not liav and not lnoc:  # force mean change to zero at zero degrees warming
    c0 = mean_changes_per_warming_level[0]
    c0[np.logical_not(np.isnan(c0))] = 0
    mean_changes_per_warming_level[0] = c0
mean_changes_per_warming_level_unfilled = mean_changes_per_warming_level.copy()

# fill data gaps in the gcm x mod matrix
stats = mean_changes_per_warming_level
means_over_gapless_mods = np.mean(stats[:,:,gapless_mods], axis=2)
stds_over_gapless_mods = np.sqrt(np.mean(np.square(stats[:,:,gapless_mods] - np.expand_dims(means_over_gapless_mods, axis=-1)), axis=2))
for g, gcm in enumerate(gcms):
    for m, mod in enumerate(mods):
        if not gaps_gcms_mods[g,m]: continue
        delta = np.zeros(ce.warming_levels.size)
        n = delta.copy()
        for k in range(len(gcms)):
            if gaps_gcms_mods[k,m]: continue
            delta_add = (stats[:,k,m] - means_over_gapless_mods[:,k]) * stds_over_gapless_mods[:,g] / stds_over_gapless_mods[:,k]
            delta_valid = np.logical_not(np.logical_or(np.isnan(delta_add), np.isinf(delta_add)))
            delta[delta_valid] = delta[delta_valid] + delta_add[delta_valid]
            n += delta_valid
        stats[:,g,m] = means_over_gapless_mods[:,g] + delta / n

if not liav and not lnoc:  # repeat force mean change to zero at zero degrees warming to get this right also for the filled data
    mean_changes_per_warming_level[0] = 0

# compute multi-model medians
mean_changes_per_warming_level_median_over_gcms = np.nanpercentile(mean_changes_per_warming_level, 50., axis=1)  # shape (ce.warming_levels.size, len(mods))
mean_changes_per_warming_level_median_over_mods = np.nanpercentile(mean_changes_per_warming_level, 50., axis=2)  # shape (ce.warming_levels.size, len(gcms))
mean_changes_per_warming_level_median_over_gcms_and_mods = np.nanpercentile(mean_changes_per_warming_level, 50., axis=(1,2))  # shape (ce.warming_levels.size)

# concatenate arrays
mean_changes_per_warming_level = \
np.concatenate((
np.expand_dims(
np.concatenate((np.expand_dims(mean_changes_per_warming_level_median_over_gcms_and_mods, axis=1), mean_changes_per_warming_level_median_over_mods), axis=1),  # shape (ce.warming_levels.size, 1+len(gcms))
axis=2),  # shape (ce.warming_levels.size, 1+len(gcms), 1)
np.concatenate((np.expand_dims(mean_changes_per_warming_level_median_over_gcms, axis=1), mean_changes_per_warming_level_unfilled), axis=1),  # shape (ce.warming_levels.size, 1+len(gcms), len(mods))
), axis=2)  # shape (ce.warming_levels.size, 1+len(gcms), 1+len(mods))



# save cube
event_category_underscore = ce.hazard_cube_name[sec]
event_category_hyphen = event_category_underscore.replace('_', '-')

indicator_underscore = ('' if lnoc else (('relative' if lrelative else 'absolute')+'_change_in_'))+('population_exposed_to_' if lpopulation else 'land_area_affected_by_')+event_category_underscore
if liav: indicator_underscore = 'interannual_standard_deviation_of_' + indicator_underscore
indicator_hyphen = indicator_underscore.replace('_', '-')
indicator_label = indicator_underscore[:1].upper() + indicator_underscore[1:].replace('_', ' ')

odir = '../cube_data/'+indicator_hyphen+'/future-projections/'+cor+'/'
ofile = indicator_hyphen+'_future-projections_'+cor+'_versus-temperature-change.nc'
opath = odir + ofile
ce.ensure_dir(opath)

ce.save2cube(
opath,
mean_changes_per_warming_level,
indicator_underscore,
indicator_underscore,
event_category_underscore,
('% of preindustrial value' if lrelative and not lnoc else '% of '+('population' if lpopulation else 'land area')),
cor, 
indicator_label,
np.array(ce.warming_levels),
None,
np.array(['multi-model-median']+gcms),
np.array(['multi-model-median']+mods))
