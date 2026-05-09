"""
Task 2b - Real Nairobi Map Colouring
CCS 2226 Foundations of AI - 2026
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from constraint import Problem
import osmnx as ox
from shapely.ops import unary_union

# ── 1. Solve CSP ──────────────────────────────────────────────────────────────
sub_counties = [
    "Westlands", "Dagoretti North", "Dagoretti South",
    "Langata", "Kibra", "Roysambu",
    "Kasarani", "Ruaraka", "Embakasi South",
    "Embakasi North", "Embakasi Central", "Embakasi East",
    "Embakasi West", "Makadara", "Kamukunji",
    "Starehe", "Mathare"
]

neighbours = [
    ("Westlands", "Dagoretti North"), ("Westlands", "Roysambu"),
    ("Westlands", "Starehe"), ("Dagoretti North", "Dagoretti South"),
    ("Dagoretti North", "Kibra"), ("Dagoretti South", "Langata"),
    ("Dagoretti South", "Kibra"), ("Langata", "Kibra"),
    ("Langata", "Embakasi West"), ("Roysambu", "Kasarani"),
    ("Roysambu", "Starehe"), ("Kasarani", "Ruaraka"),
    ("Kasarani", "Mathare"), ("Ruaraka", "Embakasi North"),
    ("Ruaraka", "Mathare"), ("Embakasi North", "Embakasi Central"),
    ("Embakasi Central", "Embakasi South"),
    ("Embakasi Central", "Embakasi East"),
    ("Embakasi East", "Embakasi South"),
    ("Embakasi West", "Embakasi Central"),
    ("Embakasi West", "Makadara"), ("Makadara", "Kamukunji"),
    ("Makadara", "Embakasi South"), ("Kamukunji", "Starehe"),
    ("Kamukunji", "Mathare"), ("Starehe", "Mathare"),
]

def solve_with_colours(n):
    colours = [f"C{i+1}" for i in range(n)]
    p = Problem()
    p.addVariables(sub_counties, colours)
    for r1, r2 in neighbours:
        p.addConstraint(lambda c1, c2: c1 != c2, (r1, r2))
    return p.getSolution()

solution = None
num_colours = 3
while solution is None:
    solution = solve_with_colours(num_colours)
    if solution is None:
        num_colours += 1

print(f"Minimum colours needed: {num_colours}")
for sc, col in sorted(solution.items()):
    print(f"  {sc}: {col}")

colour_palette = {
    "C1": "#4A90D9",
    "C2": "#E74C3C",
    "C3": "#2ECC71",
    "C4": "#F39C12",
}

# ── 2. Download Real Nairobi Sub-County Boundaries ────────────────────────────
print("\nDownloading boundaries from OpenStreetMap (1-2 mins)...\n")

rows = []
for sc in sub_counties:
    print(f"  Fetching: {sc}")
    try:
        gdf_sc = ox.geocode_to_gdf(f"{sc}, Nairobi, Kenya")
        geom = unary_union(gdf_sc.geometry)
        rows.append({"sub_county": sc, "geometry": geom})
    except Exception as e:
        print(f"    WARNING: Could not fetch {sc} — {e}")

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
gdf["colour"]     = gdf["sub_county"].map(solution)
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
    try:
        centroid = row.geometry.centroid
        ax.annotate(
            f"{row['sub_county']}\n({row['colour']})",
            xy=(centroid.x, centroid.y),
            ha="center", va="center",
            fontsize=6, fontweight="bold",
            color="white"
        )
    except Exception:
        pass

ax.set_title(
    f"Nairobi Sub-Counties — CSP Map Colouring\nMinimum Colours Used: {num_colours}",
    fontsize=14, fontweight="bold", pad=15
)
ax.axis("off")

used = sorted({solution[sc] for sc in sub_counties})
legend_handles = [
    mpatches.Patch(color=colour_palette[c], label=f"Colour {c[1:]} ({c})")
    for c in used
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("nairobi_real_map.png", dpi=150, bbox_inches="tight")
print("\nMap saved as nairobi_real_map.png")
plt.show()