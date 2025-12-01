ACC-9 (cool->warm belts) package
- Data: acc9_belts.npz (id_grid; 0=mask/white, 1..9=TT,SS,PP,AA,ST,PS,AP,PT,AS)
- GeoTIFF: acc9.tif
- Vectors: acc9.shp/*.gpkg
- Palette: palette_9_pdfcolors.csv (edit R,G,B; or use --override)
- Renderer: render_cartopy_acc9.py (use --proj robinson|mercator)


python .\render_cartopy_acc9.py --npz .\acc9_belts.npz --palette .\palette_9_pdfcolors.csv
