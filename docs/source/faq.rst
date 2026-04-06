FAQ
===

The lagged mutual information plot shows an upward trend at large lags. Is this a problem?
-------------------------------------------------------------------------------------------

This typically happens when the lag approaches the length of the recordings. Mutual information is sensitive to the number of independent data points available, and at large lags there are fewer timepoint-pairs that are far enough apart, which inflates the MI estimate. This effect is more pronounced with shorter recordings (e.g. ~10 minutes).

This is not a cause for concern. The purpose of the plot is to confirm that the data has temporal structure at longer timescales than a Markov model would predict. The gap between the real and Markov MI curves at moderate lags (e.g. a few seconds up to 20-30s) indicates the range of state durations you should expect. If you want to include this plot in a paper, consider restricting the x-axis to lags where the estimates are stable (before they start rising).

``generate_grid_movies`` threw an error about NaNs or negatives in my centroids — what's going on?
----------------------------------------------------------------------------------------------------

**Negative values in centroids**

``generate_grid_movies`` expects centroids with a **corner-of-frame origin**, where (0, 0) is a corner of the image. If you are using depth-MoSeq data, the most likely cause of negative centroids is that you are passing centroids with a **center-of-frame origin**, where (0, 0) is the center of the image. Depth-MoSeq extracts both — make sure you are using the corner-origin (px) centroids rather than the center-origin (mm) centroids.

If you are *not* using depth-MoSeq data (or you are already using the px centroids), you likely have an upstream issue. Inspect your centroids to understand where the negative values are coming from — for example, plot them over time and look for outliers or systematic offsets.

**NaN values in centroids**

NaN values typically arise from:

- **Dropped frames** during depth video collection, or
- **Frames with no detected keypoints** (if using keypoint-MoSeq).

Before calling ``generate_grid_movies``, make sure that NaNs are few and far between and don't occur in long contiguous runs (which would indicate a more serious data-quality issue). Then interpolate the NaNs using your method of choice. A simple linear interpolation with pandas works well:

.. code-block:: python

   import pandas as pd
   import numpy as np

   # centroids[key] has shape (n_timesteps, 2)
   for key in centroids:
       df = pd.DataFrame(centroids[key], columns=["x", "y"])
       df = df.interpolate(method="linear", limit_direction="both")
       centroids[key] = df.values
