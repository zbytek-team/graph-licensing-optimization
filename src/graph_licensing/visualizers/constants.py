"""Constants for graph visualization."""

# Color mapping for different license types
COLOR_MAP = {
    "solo": "#f6d700",
    "duo": "#c3102f",
    "family": "#003667",
    "default": "#000001",
    "unassigned": "#cccccc",
}

# Size mapping for different node types
SIZE_MAP = {"owner": 500, "member": 300, "solo": 400}

# Default visualization settings
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_NODE_SIZE = 500
DEFAULT_EDGE_WIDTH_UNASSIGNED = 1
DEFAULT_EDGE_WIDTH_GROUP = 3
