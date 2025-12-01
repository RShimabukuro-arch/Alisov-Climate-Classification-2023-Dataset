#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACC-27 Cartopy Renderer (Robinson)
- PDF/TIF 27-color palette
- User overrides (--override LABEL R G B)
- ONLY these graticules + labels:
    Longitudes: 90W, 180, 90E, 0
    Latitudes : 60S, 30S, EQ, 30N, 60N
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def build_palette(ordered_labels, palette_csv=None, overrides=None):
    max_id = 27
    palette = np.zeros((max_id+1, 3), dtype=np.uint8)
    palette[0,:] = [255,255,255]  # background white
    if palette_csv:
        pal = pd.read_csv(palette_csv).sort_values("id")
        for _,row in pal.iterrows():
            i = int(row["id"])
            if 0 <= i <= max_id:
                palette[i,:] = [int(row["R"]), int(row["G"]), int(row["B"])]
    else:
        import colorsys
        for i in range(1, max_id+1):
            h = (i*360.0/max_id) % 360.0
            r,g,b = colorsys.hsv_to_rgb(h/360.0, 0.75, 0.9)
            palette[i,:] = [int(255*r), int(255*g), int(255*b)]
    if overrides:
        lab_to_id = {lab: i+1 for i,lab in enumerate(ordered_labels)}
        for lab, rgb in overrides.items():
            if lab in lab_to_id:
                palette[lab_to_id[lab],:] = rgb
    return palette

def lon_formatter(x, pos=None):
    xi = int(np.round(((x + 180) % 360) - 180))  # normalize to [-180,180]
    if xi == -180 or xi == 180: return "180°"
    if xi == -90:  return "90°W"
    if xi == 90:   return "90°E"
    if xi == 0:    return "0°"
    return ""

def lat_formatter(y, pos=None):
    yi = int(np.round(y))
    if yi == -60: return "60S°"
    if yi == -30: return "30S°"
    if yi == 0:   return "EQ"
    if yi == 30:  return "30N°"
    if yi == 60:  return "60N°"
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="acc27_masked_ids_fixed.npz")
    ap.add_argument("--palette", default="palette_27_pdfcolors.csv")
    ap.add_argument("--proj", choices=["robinson","mercator"], default="robinson"); ap.add_argument("--out", default="acc27_robinson_cartopy_labeled.png")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--override", nargs=4, action="append",
                   metavar=("LABEL","R","G","B"),
                   help="override a label color, e.g., --override TdTm 139 69 19 (repeatable)")
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    ids = Z["id_grid"].astype(np.uint8)  # 0..27
    lats = Z["lats"]
    lons0 = Z["lons"]
    ordered_labels = [s for s in Z["ordered_labels"]]

    overrides = {}
    if args.override:
        for (label, r, g, b) in args.override:
            overrides[label] = (int(r), int(g), int(b))

    palette = build_palette(ordered_labels, palette_csv=args.palette, overrides=overrides)

    NY, NX = ids.shape
    # edges for pcolormesh
    lat_edges = np.linspace(-90, 90, NY+1)
    lon_edges = np.linspace(-180, 180, NX+1)
    ids_roll = np.roll(ids, -NX//2, axis=1)  # shift to -180..180 (no vertical flip)

    cmap = ListedColormap(palette/255.0)
    norm = BoundaryNorm(np.arange(0, palette.shape[0]+1)-0.5, cmap.N)

    fig = plt.figure(figsize=(12,6), dpi=args.dpi)
    ax = plt.axes(projection=ccrs.Robinson() if args.proj=="robinson" else ccrs.Mercator())
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="none", edgecolor="0.5", linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="0.3", linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="0.3", linewidth=0.3)

    # Specific graticules & labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color="gray", alpha=0.6, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.array([-180, -90, 0, 90]))
    gl.ylocator = mticker.FixedLocator(np.array([-60, -30, 0, 30, 60]))
    gl.xformatter = mticker.FuncFormatter(lon_formatter)
    gl.yformatter = mticker.FuncFormatter(lat_formatter)
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # Equator thicker line
    ax.plot(np.linspace(-180, 180, 361), np.zeros(361),
            transform=ccrs.PlateCarree(), color="k", linewidth=0.9, alpha=0.8)

    # Data
    LonE, LatE = np.meshgrid(lon_edges, lat_edges)
    pc = ax.pcolormesh(LonE, LatE, ids_roll, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="flat")
    ax.set_title("27 climatic regions (sp<775 hPa masked) — Shimabukuro et al.(2023)", fontsize=10)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
