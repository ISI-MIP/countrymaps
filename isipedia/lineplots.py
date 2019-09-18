import os
import numpy as np
import netCDF4 as nc

from isipedia.config import cube_data_stefan, cube_data_out, mask_file, totpopfile, gridareafile


class Variable:
    def __init__(self, name):
        """ e.g. land-area-affected-by-drought-absolute-changes_ISIMIP-projections_versus-temperature-change/
        """
        self.name = name
        vars(self).update(self._parse_name())

    def _parse_name(self):
        variable, studytype, axis = self.name.split('_')
        prefixes = ['land-area-affected-by', 'population-exposed-to']
        prefix = ''
        for prefix in prefixes:
            if variable.startswith(prefix):
                variable = variable[len(prefix)+1:]
                break
        suffixes = ['absolute-changes', 'relative-changes']
        suffix = ''
        for suffix in suffixes:
            if variable.endswith(suffix):
                variable = variable[:-len(suffix)-1]
                break
        return {'exposure':prefix, 'indicator':variable, 'change':suffix, 'studytype':studytype, 'axis':axis}
    
    @property
    def ncvariable(self):    
        params = vars(self).copy()
        return '{changeS}-in-{exposure}-{indicator}'.format(changeS=params.pop('change')[:-1], **params).replace('-','_')
        
    @property
    def gridnc(self):
        params = vars(self).copy()
        axis = params.pop('axis')
        axis = 'versus-year' if axis in ['versus-timeslices', 'versus-years'] else axis
        return (cube_data_stefan +
                '{variable}/future-projections/grid-level/{variable}_future-projections_grid-level_{axis}.nc'.format(
                    variable=self.ncvariable.replace('_','-'), axis=axis, **params))


    def jsonfile(self, area):
        return os.path.join(cube_data_out, self.indicator, self.studytype, area, self.name+'_'+area+'.json')
    
    @property
    def griddir(self):
        return os.path.join(cube_data_out, self.indicator, self.studytype, 'world', 'maps', self.name)

    def gridcsv(self, point):
        return os.path.join(self.griddir, str(point)+'.csv')
    
    def __repr__(self):
        return 'Variable("{}")'.format(self.name)



class Point:
    def __init__(self, scenario, climate_model, impact_model, slice):
        self.climate_model = climate_model.replace('multi-model-median','median')
        self.impact_model = impact_model.replace('multi-model-median','median')
        self.slice = slice
        self.scenario = scenario
        
    def __str__(self):
        return '{scenario}_{climate_model}_{impact_model}_{slice}'.format(**vars(self))
  
    def __repr__(self):
        return 'Point({scenario}, {climate_model}, {impact_model}, {slice})'.format(**{k:repr(v) 
                                                                                       for k, v in vars(self).items()})


class Area:
    def __init__(self, code, name=None, properties=None, mask=None, geom=None):
        self.code = code
        self.name = name
        self.properties = properties
        self.mask = mask
        self.geom = geom

    def __str__(self):
        return '{} :: {}'.format(self.code, self.name)

    def __repr__(self):
        return 'Area({}, {})'.format(self.code, self.name)


class JsonFileCreator:

    plot_label_x = ""
    plot_unit_x = ""
    plot_title = ""
    
    def __init__(self, variable, gridareafile=gridareafile, totpopfile=totpopfile):
        self.variable = variable
        self._init_axes()
        #vars(self).update(self.axes) # temperature_list etc.

        self.name = variable.ncvariable.replace('_',' ')

    def opennc(self):
        return nc.Dataset(self.variable.gridnc)
    
    def header(self, area):
        return {
             'plot_type': self.plot_type,
              "indicator": self.variable.indicator,
              "variable": self.variable.ncvariable.replace('-',' '),
              "assessment_category": self.variable.studytype,
              "area": area.code,
              "region": area.name,
              #"esgf_search_url": "https://esg.pik-potsdam.de/esg-search/search/?offset=0&limit=10&type=Dataset&replica=false&latest=true&project=ISIMIP2b&sector=Water+Global&distrib=false&replica=false&facets=world_region%2Cvariable%2Ctime_frequency%2Clicence%2Cproduct%2Cexperiment%2Cproject%2Ccountry%2Csector%2Cimpact_model%2Cperiod%2Cbias_correction%2Cdataset_type%2Cmodel%2Cvariable_long_name%2Cco2_forcing%2Csocial_forcing%2Cirrigation_forcing%2Cvegetation%2Ccrop%2Cpft%2Cac_harm_forcing%2Cdiaz_forcing%2Cfishing_forcing%2Cmf_region%2Cocean_acidification_forcing%2Cwr_station%2Cwr_basin%2Cmelting_forcing%2Cpopulation_forcing%2Cearth_model_forcing%2Cadaptation_forcing%2Cforestry_stand",
              "plot_title": self.plot_title,
              "plot_label_x": self.plot_label_x,
              "plot_unit_x": self.plot_unit_x,
              "plot_label_y": self.name,
              "plot_unit_y": "% of land area" if self.variable.exposure.startswith('land-area') else '% of population',
        }
        
    def data(self, ds, area):
        return {}
    
    def todict(self, ds, area):
        js = {}
        js.update(self.header(area))
        js.update(self.axes)
        js.update({
            "data": self.data(ds, area),
        })
        return js    
    

class Aggregator:
    def __init__(self, mask, weights):
        self.mask = mask
        self.weights = weights
        self.total_per_cell = weights[mask]
        self.total_per_area = np.sum(self.total_per_cell)

    def __call__(self, array):
        if hasattr(array, 'mask'):
            array = array.filled(np.nan)  # TODO: check missing data issues more carefully        
        res = np.sum(array[self.mask]*self.total_per_cell)/self.total_per_area
        return res.tolist() if np.isfinite(res) else None



class LandWeight:
    def __init__(self, gridareafile=gridareafile):
        with nc.Dataset(gridareafile) as ds:
            self.cell_weights0 = ds['cell_area'][:].squeeze()

    def get_weight(self, t=None):
        return self.cell_weights0


class PopolationWeight:
    def __init__(self, years=[], totpopfile=totpopfile):
        # self.cell_weight = nc.Dataset('griddata/pop_tot_2005soc_0.5deg.nc4')['pop_tot'][:].squeeze()
        # years = []
        self.cell_weight = nc.Dataset(totpopfile)['pop_tot']
        self.years = years
        self.cell_weight_indices = []

        for y in years:
            if y <= 1861:
                self.cell_weight_indices.append(0)
            elif y >= 2005:
                self.cell_weight_indices.append(144)
            else:
                self.cell_weight_indices.append(y-1861)         

    def get_weight(selt, t=None):
        if not self.cell_weight_indices:
            return self.cell_weight[-1]

        i = self.cell_weight_indices[t]
        return self.cell_weight[i]

        
class JsonFileTime(JsonFileCreator):

    plot_type = "indicator_vs_timeslices"    
    plot_label_x = "Time Slices"
    plot_unit_x = ""
    
    def __init__(self, variable):

        super().__init__(variable)
        self.plot_title =  self.name + ' vs. Time slices'

        # weighting
        if self.variable.exposure.startswith('land-area'):
            self.weighting = LandWeight()
        else:
            self.weighting = PopolationWeight(self.years)

    def _init_axes(self):
        with self.opennc() as ds:
            self.climate_model_list = ds['climate_model'][:].tolist()
            self.impact_model_list = ds['impact_model'][:].tolist()
            self.climate_scenario_list = ds['ghg_concentration_scenario'][:].tolist()    
            self.years = ds['year'][:].tolist()    

        self.axes = {
          "n_timeslices": len(self.years),
          "timeslices_list": [(y-9,y+10) for y in self.years],
          "climate_scenario_list": self.climate_scenario_list,
          "climate_model_list": self.climate_model_list,
          "impact_model_list": self.impact_model_list,            
        }

    
    def _crunch_data(self, ds, area):               
        data = {}
        for t, year in enumerate(self.years):
            weight = self.weighting.get_weight(t)
            aggregator = Aggregator(area.mask, weight)

            for k, scenario in enumerate(self.climate_scenario_list):
                for i, gcm in enumerate(self.climate_model_list):
                    for j, impact in enumerate(self.impact_model_list):
                        array = ds[self.variable.ncvariable][t, :, :, k, i, j]
                        data[(year, scenario, gcm, impact)] = aggregator(array)

        return data


    def data(self, ds, area):               

        data = self._crunch_data(ds, area)

        js = {}            
        for k, scenario in enumerate(self.climate_scenario_list):
            js[scenario] = {}

            for i, gcm0 in enumerate(self.climate_model_list):
                gcm = gcm0.replace('multi-model-median','overall')
                js[scenario][gcm] = {
                    'runs': {},
                }
                
                for j, impact in enumerate(self.impact_model_list):
                    #impact = impact.replace('multi-model-median','median')
                    js[scenario][gcm]['runs'][impact] = {
                        'mean': [data[(year, scenario, gcm0, impact)] for year in self.years]
                        #'shading_upper_border': [self.aggregate(map_) for map_ in upper[self.upper.ncvariable][:, :, :, j, i]],
                        #'shading_lower_border': [self.aggregate(map_) for map_ in lower[self.lower.ncvariable][:, :, :, j, i]],
                    }
                js[scenario][gcm].update(js[scenario][gcm]['runs'].pop('multi-model-median'))
                        
        return js

    
class JsonFileTemp(JsonFileCreator):

    plot_type = "indicator_vs_temperature"
    plot_label_x = "Global warming level"
    plot_unit_x = "°C"
    
    def __init__(self, variable):
        super().__init__(variable)
        self.plot_title =  self.name + ' vs. Global warming level'

        # weighting
        if self.variable.exposure.startswith('land-area'):
            self.weighting = LandWeight()
        else:
            self.weighting = PopolationWeight()
          
    def _init_axes(self):

        with self.opennc() as ds:
            self.climate_model_list = ds['climate_model'][:].tolist()
            self.impact_model_list = ds['impact_model'][:].tolist()
            self.temperature_list = ds['temperature_change'][:].tolist()            

        self.axes = {
              "temperature_list": self.temperature_list,
              "climate_model_list": self.climate_model_list,
              "impact_model_list": self.impact_model_list,            
        }
    

    def data(self, ds, area):
        aggregator = Aggregator(area.mask, self.weighting.get_weight())

        js = {}
        for i, gcm in enumerate(self.climate_model_list):
            gcm = gcm.replace('multi-model-median','overall')
            js[gcm] = {
                'runs': {},
            }
            for j, impact in enumerate(self.impact_model_list):
                js[gcm]['runs'][impact] = {
                    'median': [aggregator(map_) for map_ in ds[self.variable.ncvariable][:, :, :, i, j]]
                    #'shading_upper_border': [self.aggregate(map_) for map_ in upper[self.upper.ncvariable][:, :, :, j, i]],
                    #'shading_lower_border': [self.aggregate(map_) for map_ in lower[self.lower.ncvariable][:, :, :, j, i]],
                }
            js[gcm].update(js[gcm]['runs'].pop('multi-model-median'))
        return js



def jsoncreator(v):
    if v.axis == 'versus-timeslices':
        return JsonFileTime(v)
    else:
        return JsonFileTemp(v)


def generate_variables(indicators, exposures, changes, axes):
    return [Variable(exposure+'-'+indicator+'-'+change+'_ISIMIP-projections_'+axis)
                for indicator in indicators for exposure in exposures for change in changes for axis in axes]


def get_areas(geom=False, mask=False, mask_file=mask_file):
    import shapely.geometry as shg
    import fiona

    with nc.Dataset(mask_file) as ds:
        codes = sorted([v[2:] for v in ds.variables.keys() if v.startswith('m_')])

        countries = list(fiona.open('TM_WORLD_BORDERS_SIMPL-0.3/TM_WORLD_BORDERS_SIMPL-0.3.shp'))
        areas = [Area(c['properties']['ISO3'], c['properties']['NAME'], geom=shg.shape(c['geometry']) if geom else None, properties=c['properties']) 
            for c in countries if c['properties']['ISO3'] in codes]

    if mask:
        with nc.Dataset(mask_file) as ds:
            for area in areas:
                area.mask = ds['m_'+area.code][:] == 1

    return sorted(areas, key=lambda a: a.code)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    areas = get_areas()

    from config import exposures, indicators, changes, axes
    parser.add_argument('--indicators', nargs='*', default=indicators, help='%(default)s')    
    parser.add_argument('--exposure', nargs='*', default=exposures, help='%(default)s')    
    parser.add_argument('--changes', nargs='*', default=changes, help='%(default)s')    
    parser.add_argument('--axes', nargs='*', default=axes, help='%(default)s')    
    parser.add_argument('--areas', nargs='*', default=[a.code for a in areas], help='%(default)s')    
    o = parser.parse_args()

    variables = generate_variables(o.indicators, o.exposures, o.changes, o.axes)

    areas = [a for a in areas if a.code in o.areas]

    with nc.Dataset(mask_file) as ds:
        for area in areas:
            area.mask = ds['m_'+area.code][:] == 1

    for v in variables:
        print(v)
        gen = jsoncreator(v)
        for area in areas:
            js = gen.todict(area)



if __name__ == '__main__':
    main()