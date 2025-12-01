ACC-8 (seasonal airmasses) package
- Data: acc8_seasonal.npz (cool8_id, warm8_id; 0=mask/white, 1..8=Td,Tm,Sd,Sm,Pd,Pm,Ad,Am)
- GeoTIFF: acc8_cool.tif, acc8_warm.tif
- Vectors: acc8_cool.shp/*.gpkg, acc8_warm.shp/*.gpkg
- Palette: palette_8_pdfcolors.csv (edit R,G,B; or use --override)
- Renderer: render_cartopy_acc8.py (use --proj robinson|mercator)


# Robinson (Default)
python .\render_cartopy_acc8.py --npz .\acc8_seasonal.npz --palette .\palette_8_pdfcolors.csv
# Mercator
python .\render_cartopy_acc8.py --npz .\acc8_seasonal.npz --palette .\palette_8_pdfcolors.csv --proj mercator
