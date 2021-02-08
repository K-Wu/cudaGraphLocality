# Testing the Locality Optimizations: SAXPY
The process runs cascaded SAXPYs (actually y=ax) over the input array to get the output. The array is large enough to flush the L2 cache. Comparing with the baseline, partitioning is used to make the array more fit into the L2 cache. Hardware features, as listed in configurations section, are also tested in this repo.

This repo also serves as a demonstration of these tested features.

## Reasoning
* RTX 3090 VRAM: 24 GB. L2 cache 6MB. #SM 82.
* 1024 threads per block. Each performs 4 (partitioned) or 1024 assignments. (256 partitions)
* Each array takes up 2.56GB. Each partitioned array takes up 10 MB.
  * 2x and 4x partition both yield worse results.


## Configurations
* cudaGraph + rwHints
* cudaGraph + partition + rwHints
* cudaGraph + partition + l2Residency + rwHints
* cudaGraph + l2Residency + rwHints
* stream (baseline)
* stream + partition + fusion
* stream + partition + fusion+  rwHints

## Discussion
Computation takes ~60 ms consistently across all settings.
* Effective BW: 682.67 GB/s

It can be the case I misconfigured the l2 residency control.
