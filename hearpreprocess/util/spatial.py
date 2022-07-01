

def get_spatial_columns(task_config):
    valid_dict = task_config["soundata_valid_spatial_events"]
    return [dim for dim in ("azimuth", "elevation", "distance")
            if valid_dict[dim]]