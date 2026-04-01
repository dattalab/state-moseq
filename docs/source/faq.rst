FAQ
===

The lagged mutual information plot shows an upward trend at large lags. Is this a problem?
-------------------------------------------------------------------------------------------

This typically happens when the lag approaches the length of the recordings. Mutual information is sensitive to the number of independent data points available, and at large lags there are fewer timepoint-pairs that are far enough apart, which inflates the MI estimate. This effect is more pronounced with shorter recordings (e.g. ~10 minutes).

This is not a cause for concern. The purpose of the plot is to confirm that the data has temporal structure at longer timescales than a Markov model would predict. The gap between the real and Markov MI curves at moderate lags (e.g. a few seconds up to 20-30s) indicates the range of state durations you should expect. If you want to include this plot in a paper, consider restricting the x-axis to lags where the estimates are stable (before they start rising).
