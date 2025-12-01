ACC-4 (seasonal belts) package
- Data: acc4_seasonal.npz (cool4_id, warm4_id; 0=mask/white, 1..4=T,S,P,A)
- GeoTIFF: acc4_cool.tif, acc4_warm.tif
- Vectors: acc4_cool.shp/*.gpkg, acc4_warm.shp/*.gpkg
- Palette: palette_4_pdfcolors.csv (edit R,G,B; or use --override)
- Renderer: render_cartopy_acc4.py (use --proj robinson|mercator)


# Robinson (Default)
python .\render_cartopy_acc4.py --npz .\acc4_seasonal.npz --palette .\palette_4_pdfcolors.csv
# Mercator
python .\render_cartopy_acc4.py --npz .\acc4_seasonal.npz --palette .\palette_4_pdfcolors.csv --proj mercator