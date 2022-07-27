#!/usr/bin/env python3
import copy
import logging
import hearpreprocess.soundata_tau2021sse_nigens as soundata_tau2021sse_nigens
from hearpreprocess.soundata_tau2021sse_nigens import (  # noqa: F401
    ExtractMetadata,
    extract_metadata_task,
)

logger = logging.getLogger("luigi-interface")

# Copy the regular TAU 2021 SSE NIGENS config - updated here for targets
generic_task_config = copy.deepcopy(soundata_tau2021sse_nigens.generic_task_config)

generic_task_config.update({
    "task_name": "soundata_tau2021sse_nigens_xy",
    "spatial_projection": "unit_xy_disc", # none, unit_sphere, unit_xy_disc, unit_yz_disc
})
