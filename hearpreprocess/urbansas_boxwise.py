#!/usr/bin/env python3
import copy
import logging
import hearpreprocess.urbansas_pointwise as urbansas_pointwise
from hearpreprocess.urbansas_pointwise import (  # noqa: F401
    ExtractMetadata,
    extract_metadata_task,
)

logger = logging.getLogger("luigi-interface")

# Copy the regular TAU 2021 SSE NIGENS config - updated here for targets
generic_task_config = copy.deepcopy(urbansas_pointwise.generic_task_config)

generic_task_config.update({
    "task_name": "urbansas_boxwise",
    "spatial_projection": "video_azimuth_region_boxwise",
    "evaluation": ["horiz_iou_120fov_5regions_boxwise"],
})
