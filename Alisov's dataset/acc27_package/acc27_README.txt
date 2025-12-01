ACC-27 (directional) package
- Data: acc27_masked_ids_fixed.npz (0=mask/white, 1..27 = ordered labels)
- GeoTIFF: acc27_masked_ids_fixed.tif (palette embedded)
- Vectors: acc27_masked_ids_fixed.shp/*.gpkg
- Palette: palette_27_pdfcolors.csv (edit R,G,B)
- Renderer: render_cartopy_acc27.py (use --proj robinson|mercator; --override LABEL R G B)
