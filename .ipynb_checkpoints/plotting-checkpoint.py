# Plotting

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# import data
filepath = 'khyber_2000_2010_tp.csv'
df1 = pd.read_csv(filepath)
df1 = df1.drop('Unnamed: 0', axis=1)

# create 'DataArray'
df2 = df1.set_index(['lat', 'lon', 'time'])
da = df2.to_xarray()

# select a date
ds = da.isel(time=50)

# plot
plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([71, 83, 30, 38])
g = ds.tp.plot(cbar_kwargs={
        "label": "Precipitation [mm/day]",
        "extend": "neither", "pad": 0.10})
g.cmap.set_under("white")
# ax.add_feature(cf.BORDERS)
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()

