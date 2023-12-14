This software is incomplete and untested,
not suitable for any use outside of ongoing development,
with no warranty whatsoever.

The following versions reflect development progress dates.

Version 0.7.3, 2023-12-13

- Expanded TimeActivityCurve and Centroid objects
- Allow toggling enhanced cluster selection off
- Allow writing all cluster masks without --debug
- Removed warnings, so millions of fit failures are only written in --debug
- Reworked --debug vs --verbose to be more intuitive and reasonable, I hope
- Built a table of cluster centroids that's saved as csv.
- Added linear slope, intercept, and area under the curve of TACs
Version 0.7.0, 2023-12-11

- Introduced a blob detection algorithm for calculating how concise
  each cluster is.
- Allow overriding of initial cluster selection if a peak at the next
  time point is higher, and that cluster is more spatially concise.

Version 0.6.7, 2023-08-29

- Aligned loggers so info and debug output should write out correctly.
- Changed, then changed back, algorithm for selecting peak cluster
  (highest of earliest peaks rather than most frequent earliest peak)
- Harmonized all hi-res TAC creation from one TimeActivityCurve method

Version 0.5.0, 2023-05-01

- Implemented multiprocessing for bootstrapping parameters
- Implemented full (but not pretty) html report
- Implemented proper caching and retrieval of skipped volumes

Version 0.4.0, 2023-03-19

- Implemented multiprocessing for simulated annealing
- Implemented all relevant data storage inside results object
- Each module adds information to an html report section.

Version 0.3.0, 2023-01-01

- Implemented simulated annealing
- Implemented bootstrap model fitting

Version 0.2.0, 2022-10-26

- Implemented through vascular correction
- Significantly enhanced curve fitting capabilities
- When loading TACs, keep only the regions specified
- Did many comparisons with matlab output

Version 0.1.6, 2022-10-10

- Full refactor to make functions more generalist
- Addition of er176 peripheral code, not yet working
- Upgrades of dependencies to latest versions

Version 0.1.5, 2022-09-01

- Implemented Vascular Mean TAC curve fitting module

Version 0.1.4, 2022-08-28

- First version with a Docker implementation
- Implemented Partial Volume Correction module