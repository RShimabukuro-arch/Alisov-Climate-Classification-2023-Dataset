#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACC-9 climate belts (cool->warm, Cartopy Robinson), with PDF-tone colors and --override."""
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.ticker as mticker

def lonf(x,pos=None):
    xi=int(np.round(((x+180)%360)-180))
    return { -180:"180°", -90:"90°W", 0:"0°", 90:"90°E", 180:"180°"}.get(xi, "")
def latf(y,pos=None):
    yi=int(np.round(y))
    return { -60:"60S°", -30:"30S°", 0:"EQ", 30:"30N°", 60:"60N°"}.get(yi, "")

def build_palette(pal_csv, labels9, overrides):
    pal=pd.read_csv(pal_csv).sort_values("id")
    maxid=pal["id"].max(); lut=np.zeros((maxid+1,3), dtype=np.uint8)
    for _,r in pal.iterrows():
        lut[int(r["id"]),:]=[int(r["R"]),int(r["G"]),int(r["B"])]
    if overrides:
        lab2id={lab:i+1 for i,lab in enumerate(labels9)}
        for lab,(R,G,B) in overrides.items():
            if lab in lab2id: lut[lab2id[lab]]=[R,G,B]
    return lut

def draw(ax, data, lut, title):
    NY,NX=data.shape
    lat_edges=np.linspace(-90,90,NY+1); lon_edges=np.linspace(-180,180,NX+1)
    ids_roll=np.roll(data,-NX//2,axis=1)
    cmap=ListedColormap(lut/255.0); norm=BoundaryNorm(np.arange(0,lut.shape[0]+1)-0.5, cmap.N)
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="none", edgecolor="0.5", linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="0.3", linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="0.3", linewidth=0.3)
    gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle=":")
    gl.top_labels=False; gl.right_labels=False
    gl.xlocator = mticker.FixedLocator(np.array([-180,-90,0,90]))
    gl.ylocator = mticker.FixedLocator(np.array([-60,-30,0,30,60]))
    gl.xformatter = mticker.FuncFormatter(lonf); gl.yformatter = mticker.FuncFormatter(latf)
    gl.xlabel_style={"size":8}; gl.ylabel_style={"size":8}
    ax.plot(np.linspace(-180,180,361), np.zeros(361), transform=ccrs.PlateCarree(), color="k", linewidth=0.9, alpha=0.8)
    LonE,LatE=np.meshgrid(lon_edges,lat_edges)
    ax.pcolormesh(LonE,LatE,ids_roll,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),shading="flat")
    ax.set_title(title, fontsize=10)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz", default="acc9_belts.npz")
    ap.add_argument("--palette", default="palette_9_pdfcolors.csv")
    ap.add_argument("--proj", choices=["robinson","mercator"], default="robinson"); ap.add_argument("--out", default="acc9_robinson.png")
    ap.add_argument("--override", nargs=4, action="append", metavar=("LABEL","R","G","B"))
    args=ap.parse_args()

    Z=np.load(args.npz, allow_pickle=True)
    ids9=Z["id_grid"].astype(np.uint8); labels9=[s for s in Z["labels9"]]

    overrides={}
    if args.override:
        for (lab,R,G,B) in args.override:
            overrides[lab]=(int(R),int(G),int(B))

    lut=build_palette(args.palette, labels9, overrides)

    fig=plt.figure(figsize=(12,6), dpi=220)
    ax=plt.axes(projection=ccrs.Robinson() if args.proj=="robinson" else ccrs.Mercator())
    draw(ax, ids9, lut, "9 climatic zones (sp<775 hPa masked) — Shimabukuro et al. (2023)")
    fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    main()
