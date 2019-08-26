# Country maps

This is work-in-progress code. Contact me before use.
![World map](dummy/world.png?raw=true "World map")

Current SVG image of some countries (Afghanistan, France, Sudan):

![AFG](https://media.githubusercontent.com/media/ISI-MIP/countrymaps/master/countries_data/AFG/country.svg)
![FRA](https://media.githubusercontent.com/media/ISI-MIP/countrymaps/master/countries_data/FRA/country.svg)
![SDN](https://media.githubusercontent.com/media/ISI-MIP/countrymaps/master/countries_data/SDN/country.svg)

For comparison, Natural Earth mask at 1:50m resolution. Note South Sudan got removed. Click on files to see details.

![AFG](https://media.githubusercontent.com/media/ISI-MIP/countrymaps/master/countries_data_ne/AFG/country.svg)
![FRA](https://media.githubusercontent.com/media/ISI-MIP/countrymaps/master/countries_data_ne/FRA/country.svg)
![SDN](https://media.githubusercontent.com/media/ISI-MIP/countrymaps/master/countries_data_ne/SDN/country.svg)


Aims
----
- reference data for country masks as shapefile and raster, for all isimip / isipedia country-aggregation needs.
- produce SVG figures along with bounds for country maps plot on ISIPEDIA
- transform netCDF data into isipedia-compatible csv grids
- more ?


Country masks
-------------
Currently we use:
- vector file: TM_WORLD_BORDERS_SIMPL-0.3 (in zip and unzipped versions)
- raster file: CountryMask.NtoS.plusATA.nc
Note the original vector data is pretty old (i.e. do not account for political changes in the last years) and has crude resolution (few details that do not make nice SVG files)
We are currently in the process of updating these files.


Country plots:
--------------
See the [notebook](https://nbviewer.jupyter.org/urls/media.githubusercontent.com/media/ISI-MIP/countrymaps/master/maps.ipynb)


Note git-lfs
------------
I use git-lfs for jupyter notebooks, figures etc... make sure it is installed on your computer if you use git as command line and need access to these files.
