#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACC-8 seasonal airmass maps (Cartopy, Robinson/Mercator)
COOL and WARM season maps (separate PNGs)
Colors from palette_8_pdfcolors.csv (0..8 -> RGB), with optional --override
Graticules shown:
  Longitudes: 90W, 180, 90E, 0
  Latitudes : 60S, 30S, EQ, 30N, 60N
Thick Equator line.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def build_palette(labels8, palette_csv=None, overrides=None):
    max_id = 8
    palette = np.zeros((max_id+1, 3), dtype=np.uint8)
    palette[0,:] = [255,255,255]
    if palette_csv:
        pal = pd.read_csv(palette_csv).sort_values("id")
        for _,row in pal.iterrows():
            i = int(row["id"])
            if 0 <= i <= max_id:
                palette[i,:] = [int(row["R"]), int(row["G"]), int(row["B"])]
    else:
        import colorsys
        base = [(0.0,0.8,0.9),(0.02,0.8,0.9),(0.15,0.8,0.9),(0.18,0.8,0.9),(0.58,0.8,0.9),(0.60,0.5,0.8),(0.78,0.7,0.85),(0.82,0.5,0.85)]
        for i in range(1,9):
            h,s,v = base[i-1]
            r,g,b = colorsys.hsv_to_rgb(h, s, v); palette[i,:]=[int(255*r),int(255*g),int(255*b)]
    if overrides:
        lab_to_id = {lab:i+1 for i,lab in enumerate(labels8)}
        for lab, rgb in overrides.items():
            if lab in lab_to_id:
                palette[lab_to_id[lab],:] = rgb
    return palette

def lon_formatter(x, pos=None):
    xi = int(np.round(((x + 180) % 360) - 180))
    if xi in (-180, 180): return "180°"
    if xi == -90: return "90°W"
    if xi == 90:  return "90°E"
    if xi == 0:   return "0°"
    return ""

def lat_formatter(y, pos=None):
    yi = int(np.round(y))
    if yi == -60: return "60S°"
    if yi == -30: return "30S°"
    if yi == 0:   return "EQ"
    if yi == 30:  return "30N°"
    if yi == 60:  return "60N°"
    return ""

def draw_one(ax, data, palette, title):
    NY, NX = data.shape
    lat_edges = np.linspace(-90, 90, NY+1)
    lon_edges = np.linspace(-180, 180, NX+1)
    ids_roll = np.roll(data, -NX//2, axis=1)

    cmap = ListedColormap(palette/255.0); norm = BoundaryNorm(np.arange(0, palette.shape[0]+1)-0.5, cmap.N)
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="none", edgecolor="0.5", linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="0.3", linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="0.3", linewidth=0.3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle=":")
    gl.top_labels=False; gl.right_labels=False
    gl.xlocator = mticker.FixedLocator(np.array([-180,-90,0,90]))
    gl.ylocator = mticker.FixedLocator(np.array([-60,-30,0,30,60]))
    gl.xformatter = mticker.FuncFormatter(lon_formatter)
    gl.yformatter = mticker.FuncFormatter(lat_formatter)
    gl.xlabel_style={"size":8}; gl.ylabel_style={"size":8}

    ax.plot(np.linspace(-180,180,361), np.zeros(361), transform=ccrs.PlateCarree(), color="k", linewidth=0.9, alpha=0.8)
    LonE, LatE = np.meshgrid(lon_edges, lat_edges)
    ax.pcolormesh(LonE, LatE, ids_roll, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="flat")
    ax.set_title(title, fontsize=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="acc8_seasonal.npz")
    ap.add_argument("--palette", default="palette_8_pdfcolors.csv")
    ap.add_argument("--out_cool", default="acc8_ONDJFM_robinson.png")
    ap.add_argument("--out_warm", default="acc8_AMJJAS_robinson.png")
    ap.add_argument("--proj", choices=["robinson","mercator"], default="robinson")
    ap.add_argument("--override", nargs=4, action="append", metavar=("LABEL","R","G","B"))
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    cool = Z["cool8_id"].astype(np.uint8)
    warm = Z["warm8_id"].astype(np.uint8)
    labels8 = [s for s in Z["labels8"]]

    overrides = {}
    if args.override:
        for (label, r, g, b) in args.override:
            overrides[label] = (int(r), int(g), int(b))

    palette = build_palette(labels8, palette_csv=args.palette, overrides=overrides)
    proj = ccrs.Robinson() if args.proj=="robinson" else ccrs.Mercator()

    fig = plt.figure(figsize=(12,6), dpi=220)
    ax = plt.axes(projection=proj)
    draw_one(ax, cool, palette, f"8 airmasses (ONDJFM), Shimabukuro et al. (2023) ({args.proj.title()})")
    fig.savefig(args.out_cool, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(12,6), dpi=220)
    ax = plt.axes(projection=proj)
    draw_one(ax, warm, palette, f"8 airmasses (AMJJAS), Shimabukuro et al. (2023) ({args.proj.title()})")
    fig.savefig(args.out_warm, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    main()
