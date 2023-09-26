# Plotting

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from utils.config import RESULTS_DIR
from mpl_toolkits.basemap import Basemap

def simple_plot(filepath='data/uib_spatial.csv'):
    """ Simple plot without cartopy """
    # import data
    df1 = pd.read_csv(filepath)
    df1 = df1.drop('Unnamed: 0', axis=1)

    # create 'DataArray'
    df2 = df1.set_index(['lat', 'lon', 'tp'])
    da = df2.to_xarray()

    # select a date
    ds = da.isel(tp=50)

    # plot
    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    g = ds.tp.plot(cbar_kwargs={
            "label": "Precipitation [mm/day]",
            "extend": "neither", "pad": 0.10})
    g.cmap.set_under("white")
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
    

def facetgrid_plot(filepath='khyber_2000_2010_tp.csv'):
    """ Facet grid plot """
    # import data
    df1 = pd.read_csv(filepath)
    df1 = df1.drop('Unnamed: 0', axis=1)

    # create 'DataArray'
    df2 = df1.set_index(['lat', 'lon', 'time'])
    da = df2.to_xarray()
    
    # select a date range (optional)
    ds = da.isel(time=np.arange(12)) # takes 12 first time steps

    # plot
    fig = plt.figure()
    g = ds.tp.plot(col='time', col_wrap=4, cbar_kwargs={
            "label": "Precipitation [mm/day]",
            "extend": "neither"}, subplot_kws={
            "projection": ccrs.LambertConformal()})
    g.set_xlabels('Longitude °E')
    g.set_ylabels('Latitude °N')
    g.set_titles('{coord} : {value}')
    plt.show()
    
def plot_uib_etopo_image():
    
    m = Basemap(projection='lcc', resolution='l', lat_0=34.5, lon_0=77.0, width=1.3e+6, height=(1e+6)*0.8)
    
    m.bluemarble()
    m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
    parallels = np.arange(30,40,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=11)
    # draw meridians
    meridians = np.arange(72,82.5,2.5)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=11)
    


