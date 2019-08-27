# Note: Just tentatively salvaging methods for figure making.
# These will be imported by countrydataextractor.py
# checker should be included, just to be safe side.
# By Yusuke Satoh (IIASA)



#"""
#ex)                              original                                      with margin
#dict_country = [( country, [ ID, [[slat, elat, slon, elon], [sy, ey, sx, ex]], [[slat, elat, slon, elon], [sy, ey, sx, ex]], roller_flag ]), ]
#countries = ['Global', ..., 'Germany', ..., 'India', ..., 'China',... ]
#"""


def add_Global(dict_country):

    gaullist = dict_country.items()
    gaullist.insert(0, ('Global', [-999, [[-60, 90, -180, 180], [0, 299, 0, 719]], [[-60, 90, -180, 180], [0, 299, 0, 719]], False]))
    return OrderedDict(gaullist)


def gen_basemap(country, dict_country):

    id = dict_country[country][0]
    slat, elat, slon, elon = dict_country[country][2][0]
    print('{:<40}: ({}) llcrnrlat={}, urcrnrlat={}, llcrnrlon={}, urcrnrlon={}'.format(country, id, elat, slat, slon, elon))
    return Basemap(resolution='l', llcrnrlat=elat, urcrnrlat=slat, llcrnrlon=slon, urcrnrlon=elon)


def map_checker(src, title, figname, outdir, basemap=False, stop_here=False):

    if not os.path.isdir(outdir): os.makedirs(outdir)

    fig = plt.figure(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.05, top=0.95)
    ax = fig.add_subplot(111)
    ax.set_title('{}'.format(title), fontsize=6)
    if not basemap:
        im = plt.imshow(src)
    else:
        im = bm.imshow(np.flipud(src))
        bm.drawcoastlines(linewidth=0.5)
        bm.drawcountries(linewidth=0.25)
    cb = plt.colorbar(im)

    figpath = genpath(outdir, figname+'.jpg')
    plt.savefig(figpath)
    print('map_checker: {}'.format(figpath))

    if stop_here:
        sys.exit()


def drawmap(src, country, bm, outputdir, **kwargs):    # src: (ny2, nx2)

    assessment_category = kwargs['ac']
    information_level = kwargs['inflevl']
    topic = kwargs['tpc']
    impact_index = kwargs['indx']
    subindex = kwargs['subindx']
    content = kwargs['mtr']
    mtr_unit = kwargs['mtr_unit']
    threshold = kwargs['thrsh']
    temporal_type = kwargs['tt']
    scn = kwargs['scn']
    absolute_or_change = kwargs['absorchng']
    country_value_type = kwargs['cnt_val_type']
    prd = kwargs['prd']
    ensemble_stats_type = kwargs['ensem_stats_type']

    if absolute_or_change == 'abs':      unit = mtr_unit
    elif absolute_or_change == 'change': unit = '%'

    # gim mean
    fig = plt.figure(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.05, top=0.9)
    ax = fig.add_subplot(111)
    ax.set_title('{} in ensemble {} ({})\n{} for {}({})'.format(absolute_or_change, ensemble_stats_type, country_value_type,
                                                                content, impact_index, threshold),
                                                                fontsize=6)

    if absolute_or_change == 'abs':
        # TODO: [Yusuke] automatically set a better color range and a color bar for each case...
        #vmax = 1e+6
        #interval_norm = [0, 1e+3, 1e+4, 5e+4, 1e+5, 5e+5, 1e+6, 2e+6, 3e+6]
        #cmap = plt.cm.YlGnBu
        #norm = colors.BoundaryNorm(interval_norm, 256)
        #im = bm.imshow(np.flipud(src), norm=norm, cmap=cmap, vmin=0., vmax=vmax)
        if content == 'index':
            vmax = dict_impactindex[impact_index][6]
            interval_norm = dict_impactindex[impact_index][7]
            cmap = dict_impactindex[impact_index][8]
            norm = colors.BoundaryNorm(interval_norm, 256)
            im = bm.imshow(np.flipud(src), norm=norm, cmap=cmap)
        else:
            im = bm.imshow(np.flipud(src))
    elif absolute_or_change == 'change':
        absmax = 100
        interval_norm = [-100, -80, -60, -40, -20, -5, 5, 20, 40, 60, 80, 100]
        cmap = plt.cm.RdBu
        norm = colors.BoundaryNorm(interval_norm, 256)
        im = bm.imshow(np.flipud(src), norm=norm, cmap=cmap, vmin=-absmax, vmax=absmax)
    bm.drawcoastlines(linewidth=0.5)
    bm.drawcountries(linewidth=0.25)

    if country == 'Global': lat_interval = 20; lon_interval = 30
    else:                   lat_interval =  2; lon_interval =  3
    bm.drawparallels(np.arange(-90, 90.1, lat_interval), labels=[True, False, True, False], linewidth=0.2, fontsize=8)
    bm.drawmeridians(np.arange(0, 360, lon_interval), labels=[True, False, False, True], linewidth=0.2, fontsize=8)

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=8)

    cb.ax.set_ylabel('[{}]'.format(unit))
    #cb.ax.set_title(unit)

    figureinfo = '@{}\n{}\n{}'.format(country, prd, scn)
    ax.text(0.02, 0.1, figureinfo, fontsize=10, transform=ax.transAxes)

    figname = 'map_{}_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(country.replace(' ', '_'), impact_index, threshold, content, scn,
                                                          country_value_type, ensemble_stats_type, absolute_or_change, prd)
    figpath = genpath(outputdir, figname)
    plt.savefig(figpath)
    print('savefig: {}'.format(figpath))
    plt.close()


def drawplot(srcs, country, outputdir, **kwargs):    # srcs: (ngcm, ngim, nyear)
    #print('\ndrawplot for {}...'.format(country))
    # TODO: Currently, this is tempral_type = year only...

    def calc_anomaly_inpercent(src, weights):
        histmean = src[:yearwindow].mean()
        return np.divide(np.convolve(src, weights, mode='valid') - histmean, histmean) * 100    # [%]


    def maskout_nan_inf(src):
        src[where(isinf(src))] = 1e+20
        src[where(isnan(src))] = 1e+20
        return masked_equal(src, 1e+20)


    def prepare_src_and_unit(srcs, mtr_unit, absolute_or_change):

        # TODO (super important!!!): Currently, all impact indices are already climatology
        if absolute_or_change == 'abs':
            unit = mtr_unit
        #    srcs = array([[np.convolve(srcs[igcm, igim], weights, mode='valid') for igim, gim in enumerate(gims)] for igcm, gcm in enumerate(gcms)])
        elif absolute_or_change == 'change':
            unit = '%'
            #    srcs = array([[calc_anomaly_inpercent(srcs[igcm, igim], weights) for igim, gim in enumerate(gims)] for igcm, gcm in enumerate(gcms)])
            srcs_hist = srcs[:, :, :yearwindow].mean(axis=2).T   #        (ngim. ngcm)
            srcs_diff = srcs.T - srcs_hist                       # (nyear, ngim, ngcm)
            srcs = np.divide(srcs_diff, srcs_hist).T * 100   # (ngcm, ngim, nyear)
            del srcs_hist, srcs_diff
        else:
            print('Error. check kwargs[absorchng]...')
            sys.exit()

        #print('srcs has been convolved. {}'.format(srcs.shape))     # (ngcm, ngim, nyear2)
        return srcs, unit


    def get_axes_simple(ax):
        # get simple axis for a plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        return ax


    assessment_category = kwargs['ac']
    information_level = kwargs['inflevl']
    topic = kwargs['tpc']
    impact_index = kwargs['indx']
    subindex = kwargs['subindx']
    content = kwargs['mtr']
    mtr_unit = kwargs['mtr_unit']
    threshold = kwargs['thrsh']
    temporal_type = kwargs['tt']
    scn = kwargs['scn']
    absolute_or_change = kwargs['absorchng']
    country_value_type = kwargs['cnt_val_type']
    gims = kwargs['gims']

    #if not content == 'index': color = dict_metric[content][2]
    #else:                      color = dict_impactindex[impact_index][9]
    color = '#808080'

    #weights = np.ones(yearwindow) / yearwindow
    #plotyears = range(years[9], years[-10], 5)             # x-axis range 2015-2089, every 5years, as this is 20years moving average

    srcs, unit = prepare_src_and_unit(srcs, mtr_unit, absolute_or_change)  # (ngcm, nghm, nprd)

    src_25percentile = maskout_nan_inf(percentile(srcs, 25, axis=(0, 1)))
    src_75percentile = maskout_nan_inf(percentile(srcs, 75, axis=(0, 1)))
    src_25percentiles = [maskout_nan_inf(percentile(srcs[igcm], 25, axis=0)) for igcm, gcm in enumerate(gcms)]  # (ngcm, nprd)
    src_75percentiles = [maskout_nan_inf(percentile(srcs[igcm], 75, axis=0)) for igcm, gcm in enumerate(gcms)]  # (ngcm, nprd)

    xticks = [year for i, year in enumerate(years) if i%5 == 4]
    ymin = src_25percentile.min()
    ymax = src_75percentile.max()

    # get values for dots in a plot figure
    # TODO: This is temporal
    dot_years = [2006, 2030, 2055, 2080]
    iyears = [years.index(dot_year) for dot_year in dot_years]
    dots = [median(srcs[:, :, iyear]) for iyear in iyears]
    #print('picked up years to plot {}'.format(len(dots)))     # (ngcm, ngim, nyear3)

    # draw a figure
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.125, top=0.9)
    ax = fig.add_subplot(111)
    ax = get_axes_simple(ax)
    ax.set_title('{} in {}({}, {})'.format(content, impact_index, threshold, country_value_type), fontsize=8)

    ax.fill_between(years, src_25percentile, src_75percentile, facecolors=color, alpha=0.1)
    for igcm, gcm in enumerate(gcms):
        ax.fill_between(years, src_25percentiles[igcm], src_75percentiles[igcm], facecolors=dict_gcm_paras[gcm][0], alpha=0.15)

    if kwargs['absorchng'] == 'change':
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.4)

    for igcm, gcm in enumerate(gcms):
        for igim, gim in enumerate(gims):
            ax.plot(years, srcs[igcm, igim], color=dict_gcm_paras[gcm][0], linestyle='-', linewidth=0.3, markersize=0)
    for igcm, gcm in enumerate(gcms):
        ax.plot(years, median(srcs[igcm], axis=0), color=dict_gcm_paras[gcm][0], linestyle='-', linewidth=0.8, markersize=0)

    ax.plot(years, median(srcs, axis=(0, 1)), color=color, linestyle='-', linewidth=0.9, markersize=0)
    ax.scatter(dot_years, dots, c=color)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlim(xmin=years[0], xmax=years[-1])
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_ylabel('[{}]'.format(unit))

    figureinfo = '@{}\n{}\n{}'.format(country, scn, absolute_or_change)
    ax.text(0.02, 0.1, figureinfo, fontsize=10, transform=ax.transAxes)

    figname = 'plot_{}_{}_{}_{}_{}_{}_{}.png'.format(country, impact_index, threshold, content, scn, country_value_type, absolute_or_change)
    figpath = genpath(outputdir, figname)
    plt.savefig(figpath)
    print('savefig: {}'.format(figpath))
    plt.close()