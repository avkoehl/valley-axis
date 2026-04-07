from importlib.resources import files


def get_sample_data() -> dict:
    base = files("valley_axis.sample_data")
    return {
        "dem": base / "dem.tif",
        "region": base / "region.tif",
        "flowlines": base / "flowlines.gpkg",
    }
