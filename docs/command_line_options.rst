Command Line Options
====================

The ``stare`` executable is the primary interface for running the STARE pipeline.

Usage
-----

.. code-block:: bash

   stare [subject] [options]

Positional Arguments
--------------------

.. option:: subject

   The subject id. Arguments 'ID' or 'sub-ID' are equivalent.

Options
-------

.. option:: -i <path>, --input-path <path>

   The path for input files (default: ``.``)

.. option:: -o <path>, --output-path <path>

   The path for output files (default: ``.``)

.. option:: --cache-path <path>

   Fast local storage for caching interim data.

.. option:: --pet-file <path>

   Override searching through input-path with a specific file.

.. option:: --tac-file <path>

   Override searching through input-path with a specific file.

.. option:: --plasma-file <path>

   Override searching through input-path with a specific file.

.. option:: -a <int>, --axial-slices-to-clip <int>

   Axial slices to clip (default: ``0``).

.. option:: -u <str>, --pet-units <str>

   PET Units, defaults to ``kBq``.

.. option:: -t <str>, --time-units <str>

   Time Units, from TACS and/or midtimes files, default to ``min``.

.. option:: --pvc-method <str>

   PVC method, only 'single target correction' (``STC``) is supported (default: ``STC``).

.. option:: --fwhm <float>

   Full width half maximum for partial volume correction (default: ``5.9``).

.. option:: --tracer <str>

   The irreversible PET tracer used, only ``FDG`` is supported (default: ``FDG``).

.. option:: -c <int>, --vasc-corr-pct <int>

   The vascular correction percentage, as an integer from 0 to 100 (default: ``0``).

.. option:: --ignore-frames <int> [<int> ...]

   Any frames listed with this argument will be ignored.

.. option:: --regions <str> [<str> ...]

   Brain region names to be quantified in STARE.

.. option:: --bootstrap-iterations <int>

   How many bootstrapped curves shall be fit to feed the annealer? (default: ``500``).

.. option:: --annealer-iterations <int>

   How many iterations should the annealer be capped at? (default: ``5000``).

.. option:: --override-step-1-cluster <path>

   A binary mask may be used in place of k-means clustering.

.. option:: --override-step-2-cluster <path>

   A binary mask may be used in place of k-means clustering.

.. option:: --resample-for-clustering <str>

   Down-sample the PET images for k-means clustering. 
   'x2' down-samples by halving in each dimension. 
   '2mm' resamples to a resolution of 2mm isotropic. 
   '3mm' resamples to a resolution of 3mm isotropic. 
   '4mm' resamples to a resolution of 4mm isotropic.

.. option:: --reduce-step-one-sparsity <int>

   The threshold for removing small blobs within the best cluster. A threshold of 10 will remove up to 10% of the voxels, from the smallest blobs, leaving the largest, most contiguous blobs, constituting at least 90% of the voxels (default: ``0``).

.. option:: --consider-alternate-step-one-cluster

   Set this to True to cause STARE to assess the step one k-means cluster, compare it to alternative clusters, and recommend an alternate selection if it finds a better option. Note that 'better' is subject to many factors and may change version-to-version. And this doesn't actually select the new cluster, but only recommends it. To implement the new cluster, use the ``--override-step-one-cluster`` option with the suggested cluster from this run.

.. option:: --keep-confetti-patterns-step-1

   By default, STARE will not consider k-means clusters that look like 'confetti on the floor'. Noisy scans may produce these and they can have high early peaks that prevent selection of good vascular clusters. This option causes STARE to include these clusters as 'likely_vascular' while scoring them.

.. option:: --drop-confetti-patterns-step-2

   By default, STARE will assume the step 1 filter excluded problematic clusters, so this is no longer needed at step 2. This option causes STARE to apply the filter again, excluding any of the four sub-clusters marked as noise.

.. option:: --latest-usable-volume <int>

   Run STARE only on the earliest N volumes specified. This can be useful for time stability analyses (default: ``-1``).

.. option:: --decompose-components

   Turn on to generate PCA and ICA component maps. Stare_pet doesn't use these yet, but they can be compared with k-means clusters.

.. option:: --save-all-cluster-masks

   Turn on to save nifti masks of all clusters, not just best. These masks will be saved in the 'debug/masks/' directory.

.. option:: --save-all-failures

   This feature is not yet available. In a future version, Turn on to save parameters of failed curve fits. This can be useful for debugging or investigating the range of parameters useful for fitting curves, but can consume tens of gigabytes of memory for hard-to-fit TACs. These failures will be written to a csv file in the 'debug/' directory.

.. option:: --ignore-spatial-info

   By default, spatial info is used to classify and remove step-one k-means clusters made up of only noise in inferior slices. Optionally, it can also be used for sparsity reduction or to consider alternate clusters. This flag can be used to turn off the calculation of spatial information and speed up processing slightly. This will prevent exclusion of vascular clusters with all noise in inferior slices. This flag will be overridden by --reduce-step-one-sparsity > 0, --consider-alternate-step-one-cluster, or --drop-confetti-patterns-step-2, because they all require spatial information.

.. option:: --stop-after-clustering

   Exit after clustering is complete.

.. option:: --stop-after-pvc

   Exit after partial volume correction is complete.

.. option:: --output-all-fit-failures

   Turn on to emit a note on each curve fit failure to the log.

.. option:: -f <str>, --options-file <str>

   A file containing command-line arguments. The arguments in the file will override defaults, but be overridden by explicit command-line arguments.

.. option:: -v, --verbose

   Set from 0 to 2 times to trigger more verbose output.

.. option:: --debug

   Log extra data and pickle extra data to the debug directory.

.. option:: --force

   Even if data are cached, recalculate and overwrite all output.

.. option:: --num-cpus <str>

   Where parallel processing is supported, use this many processes. (default: ``1``, use ``max`` for all available CPUs).

.. option:: --version

   If specified, print the version and exit.
