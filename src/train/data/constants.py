"""Sequence length and provisional log-clock scaling constants."""
MAX_PLIES = 128
# Estimated, not fitted: log1p(seconds), shared by training and inference.
CLOCK_LOG_MEAN = 4.0
CLOCK_LOG_STD = 1.2
