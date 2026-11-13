
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import rioxarray

from scipy.ndimage import generic_filter
from .altimetry_extractors import extract_altimetry_data, discover_altimetry_files

pygmt.config(GMT_COMPATIBILITY="6")


# ==========================================================
# RESOLUTION CHECK
# ==========================================================
def is_high_resolution(filename: str) -> bool:
    f = filename.upper()
    return (
        "ENHANCED_MEASUREMENT" in f or
        ("S6" in f and "__HR_" in f)
    )


# ==========================================================
# VECTOR LOADER
# ==========================================================
def load_vector(input_obj):
    gdf = (
        input_obj.copy()
        if isinstance(input_obj, gpd.GeoDataFrame)
        else gpd.read_file(input_obj)
    )
    return gdf.to_crs("EPSG:4326")


# ==========================================================
# INWARD BUFFER (2 km, MAX EXTENT)
# ==========================================================
def inward_buffer_km(gdf, buffer_km):
    lon, _ = gdf.geometry.union_all().centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    gdf_utm = gdf.to_crs(utm_crs)

    buffered = gdf_utm.buffer(-buffer_km * 1000)
    buffered = buffered[~buffered.is_empty]

    return gpd.GeoDataFrame(
        geometry=buffered,
        crs=utm_crs
    ).to_crs("EPSG:4326")


# ==========================================================
# TERRAIN HELPERS
# ==========================================================
def horn_slope_deg(z, dx, dy):
    zp = np.pad(z, 1, constant_values=np.nan)

    dzdx = (
        (zp[:-2, 2:] + 2*zp[1:-1, 2:] + zp[2:, 2:]) -
        (zp[:-2, :-2] + 2*zp[1:-1, :-2] + zp[2:, :-2])
    ) / (8 * dx)

    dzdy = (
        (zp[2:, :-2] + 2*zp[2:, 1:-1] + zp[2:, 2:]) -
        (zp[:-2, :-2] + 2*zp[:-2, 1:-1] + zp[:-2, 2:])
    ) / (8 * dy)

    return np.degrees(
        np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    )


def riley_tri(z):
    def tri(w):
        c = w[4]
        n = np.delete(w, 4)
        n = n[np.isfinite(n)]

        return (
            np.sqrt(np.sum((c - n)**2))
            if np.isfinite(c) and len(n) >= 5
            else np.nan
        )

    return generic_filter(
        z,
        tri,
        size=3,
        mode="constant",
        cval=np.nan
    )


# ==========================================================
# LAND-ONLY BUFFER FROM ALTIMETRY OBSERVATIONS
#
# BUFFER DIAMETER = 4 km
# BUFFER RADIUS   = 2 km
#
# EXCLUDE MAXIMUM RESERVOIR WATER EXTENT
# ==========================================================
def buffer_from_all_tracks(
    points_inside_water,
    max_water_gdf,
    buffer_km
):
    lon, _ = points_inside_water.geometry.union_all().centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    points_utm = points_inside_water.to_crs(utm_crs)
    water_utm = max_water_gdf.to_crs(utm_crs)

    # Buffer each altimetry observation.
    point_buffers = points_utm.geometry.buffer(
        buffer_km * 1000
    )

    # Combine overlapping buffers.
    combined_buffer = point_buffers.union_all()

    # Exclude the complete maximum water extent.
    water_polygon = water_utm.geometry.union_all()

    buffer_land = combined_buffer.difference(
        water_polygon
    )

    buffer_land = buffer_land.buffer(0)

    return gpd.GeoDataFrame(
        geometry=[buffer_land],
        crs=utm_crs
    ).to_crs("EPSG:4326")


# ==========================================================
# TERRAIN METRICS
# ==========================================================
def compute_terrain_indices_from_buffer(buffer_geom):
    geom = buffer_geom.to_crs("EPSG:4326")
    bounds = geom.total_bounds

    region = [
        bounds[0] - 0.1,
        bounds[2] + 0.1,
        bounds[1] - 0.1,
        bounds[3] + 0.1,
    ]

    dem = (
        pygmt.datasets.load_earth_relief(
            "03s",
            region=region
        )
        .rio.write_crs("EPSG:4326")
        .load()
    )

    # Only the land remaining after water exclusion
    # is used in the terrain calculations.
    dem_clip = dem.rio.clip(
        geom.geometry,
        geom.crs
    )

    z = dem_clip.values.astype(float)

    tri = riley_tri(z)

    return {
        "elev_p95_p05_m": float(
            np.nanpercentile(z, 95) -
            np.nanpercentile(z, 5)
        ),
        "tri_p95_m": float(
            np.nanpercentile(tri, 95)
        ),
    }


# ==========================================================
# CALCULATE TERRAIN VALUES
# VALUES ARE RETAINED IN THE OUTPUT
# ==========================================================
def compute_terrain_values(
    base_folder,
    max_water_input
):
    max_water = load_vector(max_water_input)
    all_points = []

    for f in discover_altimetry_files(base_folder):
        df = extract_altimetry_data(f)

        if df.empty:
            continue

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df.longitude,
                df.latitude
            ),
            crs="EPSG:4326"
        )

        all_points.append(gdf)

    if not all_points:
        return pd.DataFrame(
            columns=[
                "elev_p95_p05_m",
                "tri_p95_m"
            ]
        )

    pts_all = gpd.GeoDataFrame(
        pd.concat(
            all_points,
            ignore_index=True
        ),
        crs="EPSG:4326"
    )

    # Select altimetry observations located within
    # the maximum reservoir water extent.
    inside = gpd.sjoin(
        pts_all,
        max_water,
        predicate="intersects",
        how="inner"
    )

    if inside.empty:
        return pd.DataFrame(
            columns=[
                "elev_p95_p05_m",
                "tri_p95_m"
            ]
        )

    # Radius = 2 km, diameter = 4 km.
    # Maximum water extent is excluded inside
    # buffer_from_all_tracks().
    buffer_land = buffer_from_all_tracks(
        inside,
        max_water,
        2
    )

    if buffer_land.empty or buffer_land.geometry.is_empty.all():
        return pd.DataFrame(
            columns=[
                "elev_p95_p05_m",
                "tri_p95_m"
            ]
        )

    metrics = compute_terrain_indices_from_buffer(
        buffer_land
    )

    # Retain numerical values only.
    # They do not affect classification.
    return pd.DataFrame([metrics])


# ==========================================================
# TRACK GEOMETRY CHECK (1/2 vs 3/4)
# ==========================================================
def check_altimetry_tracks(
    base_folder,
    max_water_input,
    perm_water_input
):
    max_water = load_vector(max_water_input)
    perm_water = load_vector(perm_water_input)

    max_inner = inward_buffer_km(max_water, 2)

    records = []

    for f in discover_altimetry_files(base_folder):
        try:
            df = extract_altimetry_data(f)

            if df.empty:
                continue

            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(
                    df.longitude,
                    df.latitude
                ),
                crs="EPSG:4326"
            )

            max_hits = gpd.sjoin(
                gdf,
                max_water,
                predicate="intersects",
                how="inner"
            )

            if max_hits.empty:
                continue

            perm_hits = gpd.sjoin(
                max_hits.drop(
                    columns="index_right",
                    errors="ignore"
                ),
                perm_water,
                predicate="intersects",
                how="inner"
            )

            inner_hits = gpd.sjoin(
                perm_hits.drop(
                    columns="index_right",
                    errors="ignore"
                ),
                max_inner,
                predicate="intersects",
                how="inner"
            )

            records.append({
                "file": os.path.basename(f),
                "passes": (
                    (not perm_hits.empty) and
                    (not inner_hits.empty)
                )
            })

        except Exception:
            continue

    return pd.DataFrame(
        records,
        columns=["file", "passes"]
    )


# ==========================================================
# MASTER DRIVER
# TERRAIN VALUES RETAINED IN OUTPUT
# ==========================================================
def run_full_classification(
    base_folder,
    max_water_input,
    perm_water_input
):

    has_hr = any(
        is_high_resolution(os.path.basename(f))
        for f in discover_altimetry_files(base_folder)
    )

    track_results = check_altimetry_tracks(
        base_folder,
        max_water_input,
        perm_water_input
    )

    if track_results.empty:
        raise ValueError(
            "No altimetry observations intersect the "
            "maximum reservoir water extent."
        )

    # Classification uses only geometry and mode.
    base_class = (
        1 if has_hr else 2
    ) if track_results["passes"].all() else (
        3 if has_hr else 4
    )

    # Terrain calculations are independent
    # of the classification.
    terrain_values = compute_terrain_values(
        base_folder,
        max_water_input
    )

    # Return numerical class (1, 2, 3, or 4),
    # geometry results, and terrain values.
    return base_class, track_results, terrain_values
