"""
Task 2a - Australia Map Colouring
CCS 2226 Foundations of AI - 2026
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from constraint import Problem
import osmnx as ox

# ── 1. Solve CSP ──────────────────────────────────────────────────────────────
regions = [
    "Western Australia",
    "Northern Territory",
    "South Australia",
    "Queensland",
    "New South Wales",
]

neighbours = [
    ("Western Australia", "Northern Territory"),
    ("Western Australia", "South Australia"),
    ("Northern Territory", "South Australia"),
    ("Northern Territory", "Queensland"),
    ("South Australia", "Queensland"),
    ("South Australia", "New South Wales"),
    ("Queensland", "New South Wales"),
]

colours = ["Blue", "Red", "Green"]

p = Problem()
p.addVariables(regions, colours)
for r1, r2 in neighbours:
    p.addConstraint(lambda c1, c2: c1 != c2, (r1, r2))

solution = p.getSolution()
print("Solution:")
for region, colour in sorted(solution.items()):
    print(f"  {region}: {colour}")

colour_palette = {
    "Blue":  "#4A90D9",
    "Red":   "#E74C3C",
    "Green": "#2ECC71",
}

# ── 2. Download Real Australia Region Boundaries ──────────────────────────────
print("\nDownloading boundaries from OpenStreetMap (1-2 mins)...\n")

rows = []
for region in regions:
    print(f"  Fetching: {region}")
    gdf_r = ox.geocode_to_gdf(f"{region}, Australia")
    rows.append({"region": region, "geometry": gdf_r.geometry.iloc[0]})

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
gdf["colour"]     = gdf["region"].map(solution)
gdf["colour_hex"] = gdf["colour"].map(colour_palette).fillna("#CCCCCC")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

gdf.plot(
    ax=ax,
    color=gdf["colour_hex"],
    edgecolor="black",
    linewidth=1.5
)

for _, row in gdf.iterrows():
    centroid = row.geometry.centroid
    ax.annotate(
        f"{row['region']}\n({row['colour']})",
        xy=(centroid.x, centroid.y),
        ha="center", va="center",
        fontsize=7, fontweight="bold",
        color="white"
    )

ax.set_title(
    "Australia Regions — CSP Map Colouring\nColours: Blue, Red, Green",
    fontsize=14, fontweight="bold", pad=15
)
ax.axis("off")

legend_handles = [
    mpatches.Patch(color=colour_palette[c], label=c)
    for c in colours
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("australia_map.png", dpi=150, bbox_inches="tight")
print("\nMap saved as australia_map.png")
plt.show()