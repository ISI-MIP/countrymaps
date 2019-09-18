cube_data_stefan = '/p/tmp/slange/counting_extremes/cube_data/'
#cube_data_out = '/p/projects/isipedia/perrette/cube_data/'
cube_data_out = '/p/tmp/perrette/isipedia/cube_data/'
#mask_file = 'naturalearth_50m_countrymask.nc'
mask_file = 'CountryMask.NtoS.plusATA.nc'

# to normalize pct (sum-up over grid cells)
#landarea = np.ones((360, 720))
#population = np.ones((360, 720))
#popdir = '/p/tmp/slange/counting_extremes/input_data/population/'
griddata = 'griddata/'
#totpopfile = griddata+'pop_tot_2005soc_0.5deg.nc4'
#ruralpopfile = griddata+'pop_rural_2005soc_0.5deg.nc4'
totpopfile = griddata+'pop_tot_histsoc_0.5deg_annual_1861_2005'
# ruralpopfile = griddata+'pop_rural_2005soc_0.5deg.nc4'
gridareafile = griddata+'gridarea.nc'


indicators = [
     'drought',
     'heatwave',
     'river-flood',
     'tropical-cyclone',
#     'water-shortage',
     'crop-failure',
     'wildfire',
]

exposures = [
    'land-area-affected-by', 
    'population-exposed-to',
]

changes = [
    'absolute-changes', 
    'relative-changes',  
]

axes = [
    'versus-temperature-change',
    'versus-timeslices',
]