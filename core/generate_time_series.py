def generate_altimetry_timeseries(
    nc_folder,
    max_gdf,
    reservoir_class,
    output_dir,
    s1_search_days=30,
):
    """
    Generate a reservoir water-level time series.

    Classes:
      1, 2: Reservoir interior
      3, 4: Shoreline or seasonally inundated regions

    Water-level observation:
      - Calculate the median elevation of retained footprints.
      - Select the observed footprint closest to that median.

    Filtering:
      - Global raw MAD filter (threshold = 6.0)

    Sentinel-1 metadata:
      - s1_date: Selected Sentinel-1 acquisition date
      - s1_gap_days: Absolute temporal difference in days

    """

    # --------------------------------------------------
    # Imports
    # --------------------------------------------------
    import os
    import glob
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import ee
    import geemap

    from datetime import timedelta
    from skimage.filters import threshold_otsu

    from .altimetry_extractors import extract_altimetry_data

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    MEDIAN_CSV = os.path.join(output_dir, "MED.csv")

    # --------------------------------------------------
    # CLASS PARSER
    # FOUR CLASSES ONLY: 1, 2, 3, 4
    # --------------------------------------------------
    def parse_reservoir_class(res_class):

        if isinstance(res_class, (int, np.integer)):
            base_class = int(res_class)

        elif isinstance(res_class, str):
            value = res_class.strip()

            if value not in ["1", "2", "3", "4"]:
                raise ValueError(
                    f"Invalid reservoir_class: {res_class}. "
                    "Expected Class 1, 2, 3, or 4."
                )

            base_class = int(value)

        else:
            raise ValueError(
                f"Invalid reservoir_class: {res_class}. "
                "Expected Class 1, 2, 3, or 4."
            )

        if base_class not in [1, 2, 3, 4]:
            raise ValueError(
                f"Invalid reservoir_class: {res_class}. "
                "Expected Class 1, 2, 3, or 4."
            )

        return base_class

    base_class = parse_reservoir_class(
        reservoir_class
    )

    # Sentinel-1 mask applies only to Classes 3 and 4.
    use_s1 = base_class in [3, 4]

    # 2 km inward buffer only for Classes 1 and 2.
    buffer_m = -2000 if base_class in [1, 2] else 0

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def buffer_gdf(gdf, buffer_m):

        g = gdf.to_crs(3857)

        g["geometry"] = g.geometry.buffer(
            buffer_m
        )

        g = g[~g.is_empty]

        return g.to_crs(4326)

    def clean_sjoin_columns(gdf):

        drop_cols = [
            c for c in ["index_left", "index_right"]
            if c in gdf.columns
        ]

        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)

        return gdf

    # --------------------------------------------------
    # GLOBAL RAW MAD FILTER
    #
    # Manuscript criterion:
    #
    # |xi - median(x)| > 6 * MAD
    #
    # MAD = median(|xi - median(x)|)
    #
    # No modified z-score scaling.
    # --------------------------------------------------
    def mad_filter(df, col, thr=6.0):

        x = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        median = x.median()

        mad = np.median(
            np.abs(x - median)
        )

        if np.isnan(mad):
            return df.copy()

        # Retain observations satisfying:
        # |xi - median(x)| <= 6 * MAD

        keep = (
            np.abs(x - median) <= thr * mad
        )

        return df.loc[keep].copy()

    # --------------------------------------------------
    # SENTINEL-1 MASK
    #
    # Return:
    #   water_mask
    #   s1_date
    #   s1_gap_days
    # --------------------------------------------------
    def sentinel1_mask(region_ee, target_date):

        try:
            ee.Initialize(project='ee-abhilashagarkoti')
        except Exception:
            ee.Authenticate()
            ee.Initialize(project='ee-abhilashagarkoti')

        start = ee.Date(target_date - timedelta(days=s1_search_days))
        end   = ee.Date(target_date + timedelta(days=s1_search_days))

        coll = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region_ee)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filterDate(start, end)
            .select("VV")
        )

        if coll.size().getInfo() == 0:
            return None, None, None

        target = ee.Date(target_date)

        def add_time_diff(img):
            diff = img.date().difference(target, "day").abs()
            return img.set("time_diff", diff)

        closest_img = ee.Image(coll.map(add_time_diff).sort("time_diff").first())
        closest_date = ee.Date(closest_img.get("system:time_start"))

        same_day = coll.filterDate(closest_date, closest_date.advance(1, "day"))
        mosaic = same_day.mosaic().clip(region_ee)
        vv = mosaic.select("VV")

        # Otsu threshold
        sample = vv.sample(region=region_ee, scale=30, numPixels=5000, geometries=False)
        arr = np.array(sample.aggregate_array("VV").getInfo())
        arr = arr[np.isfinite(arr)]

        if arr.size < 100:
            return None, None, None

        thr = threshold_otsu(arr)
        water_mask = vv.lt(thr).rename("water")

        s1_date = closest_date.format("YYYY-MM-dd").getInfo()
        s1_gap_days = float(closest_img.get("time_diff").getInfo())
        return water_mask, s1_date, s1_gap_days

    # --------------------------------------------------
    # INIT EE
    # --------------------------------------------------
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # --------------------------------------------------
    # BUFFERED MAXIMUM WATER EXTENT
    # --------------------------------------------------
    buffered_max = buffer_gdf(
        max_gdf,
        buffer_m
    )

    max_union = (
        buffered_max.geometry.union_all()
    )

    region_ee = geemap.geopandas_to_ee(
        buffered_max
    )

    # --------------------------------------------------
    # OUTPUT CONTAINER
    # --------------------------------------------------
    ts_median = []

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    for nc in sorted(
        glob.glob(
            os.path.join(
                nc_folder,
                "**/*.nc"
            ),
            recursive=True
        )
    ):

        try:
            df = extract_altimetry_data(nc)

        except Exception:
            continue

        required = {
            "latitude",
            "longitude",
            "date",
            "mission",
            "altitude",
            "range"
        }

        if not required.issubset(df.columns):
            continue

        # --------------------------------------------------
        # Fill missing corrections
        # --------------------------------------------------
        for c in [
            "iono",
            "dry",
            "wet",
            "pole",
            "solid",
            "load"
        ]:

            if c not in df.columns:
                df[c] = 0.0

            df[c] = df[c].fillna(0.0)

        if "geoid" not in df.columns:
            df["geoid"] = 0.0

        df["geoid"] = (
            df["geoid"].fillna(0.0)
        )

        # --------------------------------------------------
        # Geophysical corrections
        # --------------------------------------------------
        corr = (
            df["dry"]
            + df["wet"]
            + df["pole"]
            + df["solid"]
            + df["iono"]
            + df["load"]
        )

        # --------------------------------------------------
        # Orthometric elevation
        # --------------------------------------------------
        df["elevation"] = (
            df["altitude"]
            - (df["range"] + corr)
            - df["geoid"]
        )

        df = df.dropna(
            subset=[
                "elevation",
                "latitude",
                "longitude",
                "date",
                "cycle"
            ]
        )

        # --------------------------------------------------
        # Convert to GeoDataFrame
        # --------------------------------------------------
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df.longitude,
                df.latitude
            ),
            crs="EPSG:4326"
        )

        # --------------------------------------------------
        # Retain footprints inside the selected extent.
        # --------------------------------------------------
        gdf = gdf[
            gdf.geometry.within(
                max_union
            )
        ]

        if gdf.empty:
            continue

        # --------------------------------------------------
        # Default Sentinel-1 metadata
        #
        # Remains empty for Classes 1 and 2.
        # --------------------------------------------------
        s1_date = None
        s1_gap_days = np.nan

        # --------------------------------------------------
        # SENTINEL-1 WATER MASK
        # CLASSES 3 AND 4 ONLY
        # --------------------------------------------------
        if use_s1:

            altimetry_date = pd.to_datetime(
                gdf["date"].iloc[0]
            )

            # Find nearest Sentinel-1 acquisition
            # and its temporal difference.
            mask, s1_date, s1_gap_days = (
                sentinel1_mask(
                    region_ee,
                    altimetry_date
                )
            )

            # No Sentinel-1 data: skip file.
            if mask is None:
                continue

            # --------------------------------------------------
            # Convert footprints to Earth Engine.
            # --------------------------------------------------
            fc = geemap.geopandas_to_ee(
                gdf
            )

            # --------------------------------------------------
            # Water fraction extractor
            # --------------------------------------------------
            def add_water_fraction(feat):

                geom = feat.geometry()

                result = mask.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom.buffer(100),
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e6
                )

                # Force property existence even if null.
                frac = ee.Algorithms.If(
                    result.contains("water"),
                    result.get("water"),
                    -1
                )

                return feat.set(
                    "water_frac",
                    frac
                )

            # --------------------------------------------------
            # Apply water fraction calculation.
            # --------------------------------------------------
            fc = fc.map(
                add_water_fraction
            )

            # --------------------------------------------------
            # Convert back to GeoDataFrame.
            # --------------------------------------------------
            gdf = geemap.ee_to_gdf(
                fc
            )

            # --------------------------------------------------
            # Check water fraction availability.
            # --------------------------------------------------
            if "water_frac" not in gdf.columns:
                continue

            gdf["water_frac"] = pd.to_numeric(
                gdf["water_frac"],
                errors="coerce"
            )

            # Remove missing values.
            gdf.loc[
                gdf["water_frac"] < 0,
                "water_frac"
            ] = np.nan

            gdf = gdf.dropna(
                subset=["water_frac"]
            )

            if gdf.empty:
                continue

            # --------------------------------------------------
            # Water classification
            # --------------------------------------------------
            gdf["water_flag"] = (
                gdf["water_frac"] >= 0.4
            ).astype(int)

            # Retain Sentinel-1 metadata.
            gdf["s1_date"] = s1_date
            gdf["s1_gap_days"] = s1_gap_days

            # --------------------------------------------------
            # Keep only water-covered footprints.
            # --------------------------------------------------
            gdf = gdf[
                gdf["water_flag"] == 1
            ]

            if gdf.empty:
                continue

        # --------------------------------------------------
        # Clean spatial join columns
        # --------------------------------------------------
        gdf = clean_sjoin_columns(
            gdf
        )

        mission = (
            gdf["mission"].iloc[0]
        )

        date = pd.to_datetime(
            gdf["date"].iloc[0]
        )

        # --------------------------------------------------
        # MEDIAN WATER-LEVEL OBSERVATION
        # --------------------------------------------------

        # Step 1: Median of retained elevations.
        med = gdf["elevation"].median()

        # Step 2: Find observed footprint closest
        # to the median.
        idx = (
            gdf["elevation"] - med
        ).abs().idxmin()

        pt = gdf.loc[idx]

        # Step 3: Representative observed elevation.
        e_rep = pt["elevation"]

        # Step 4: Water-level variability (MAD) of retained footprints.
        mad_pass = (gdf["elevation"] - med).abs().median()

        # --------------------------------------------------
        # Save observation and Sentinel-1 metadata.
        # --------------------------------------------------
        ts_median.append({

            "mission": mission,

            "date": date,

            "cycle": pt["cycle"],

            # Representative observation
            "elevation": e_rep,

            # Representative coordinates
            "latitude": pt["latitude"],
            "longitude": pt["longitude"],

            # Water-level variability for this pass
            "water_level_variability": mad_pass,

            # Sentinel-1 metadata
            "s1_date": s1_date,
            "s1_gap_days": s1_gap_days,
        })

    # --------------------------------------------------
    # SAVE OUTPUT WITH FILTERING
    # --------------------------------------------------
    if ts_median:

        ts = (
            pd.DataFrame(ts_median)
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")
        )

        # Apply the manuscript's raw 6 MAD criterion.
        ts = mad_filter(
            ts,
            "elevation",
            thr=6.0
        )

        ts.reset_index().to_csv(
            MEDIAN_CSV,
            index=False,
            float_format="%.3f"
        )

    return MEDIAN_CSV
