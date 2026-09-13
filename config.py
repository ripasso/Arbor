"""
Tunable constants for the Branchseed pipeline.

Keep every "magic number" here so thresholds can be tuned in one place
without hunting through the pipeline modules.
"""

# --- Stage B: search shell ---
SEARCH_SHELL_MARGIN_MM = 15.0        # how far to dilate the aorta mask outward

# --- Stage C: vesselness (Frangi) ---
FRANGI_SCALES_MM = [1.0, 1.5, 2.0, 3.0]   # sigma values for multi-scale Hessian
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_C = 500  # start high for CT HU range; tune against dev set

# --- Stage C: region growing (primary detector) ---
# hu_low / hu_high are NOT hardcoded — they come from per-case calibration
# (see region_growing.calibrate_hu_range). These are fallback/sanity bounds only.
HU_FALLBACK_LOW = 100
HU_FALLBACK_HIGH = 1500

# --- Stage E: ostium / crop-face rejection ---
CROP_FACE_EDGE_SLICES = 2        # how many extreme z-slices count as "cropped end"
PLANARITY_FLATNESS_RATIO = 0.15  # lambda3/lambda1 threshold below which a patch is "flat"

# --- Stage F: eligibility & tracing ---
ELIGIBILITY_MIN_TRACE_MM = 5.0
PROXIMAL_MAX_TRACE_MM = 10.0

# --- Ostium matching tolerance for local validation ---
MATCH_TOLERANCE_MM = 5.0

# --- Performance ---
MAX_RUNTIME_SEC_PER_CASE = 60
