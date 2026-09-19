DamBurst -> Google Earth Engine
================================

Run: bhakra-1789836180

1. Open https://code.earthengine.google.com
2. Paste the contents of damburst_gee.js into the editor.
3. Press Run.

No asset upload and no authentication beyond your normal Earth Engine login is
needed: the modelled flood extent is embedded in the script.

What you are looking at
-----------------------
Every raster layer is Earth Engine's own copy of the data -- Copernicus GLO-30,
Sentinel-2 L2A, Sentinel-1 GRD, ESA WorldCover. The only DamBurst output is the
red flood outline. That makes this an independent check: the model was built on
data fetched from AWS and the Microsoft Planetary Computer, so if the extent
lines up with GEE's terrain and imagery, two separate copies of the world agree.

One caveat worth knowing: GEE's COPERNICUS/S1_GRD is not the same product as
the sentinel-1-rtc collection used for the validation score in this run. It is
terrain-flattened differently, so the water mask in the script is a visual
sanity layer only. The number in the Validation tab comes from the RTC product.

flood_extent_ee.geojson is the same geometry as a standalone file, if you would
rather upload it as a table asset.
