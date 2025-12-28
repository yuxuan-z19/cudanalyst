# DSLs/Compilers benchmarks

## PolyBench-ACC (poly)

[cavazos-lab/PolyBench-ACC](https://github.com/cavazos-lab/PolyBench-ACC), an actively maintained fork of [sgrauerg/polybenchGpu](https://github.com/sgrauerg/polybenchGpu), is a collection of [the Polyhedral Benchmark suite](https://www.cs.colostate.edu/~pouchet/software/polybench/) codes (plus convolution) implemented for processing on the GPU using CUDA, OpenCL, and HMPP. Compared to the original polybenchGpu, PolyBench-ACC updates and standardizes the datasize configurations to better reflect modern GPU evaluation settings.

```bibtex
@inproceedings{poly,
    title        = {Auto-tuning a high-level language targeted to GPU codes},
    author       = {
        Grauer-Gray, Scott and Xu, Lifan and Searles, Robert and
        Ayalasomayajula, Sudhee and Cavazos, John
    },
    year         = 2012,
    booktitle    = {2012 Innovative Parallel Computing (InPar)},
    volume       = {},
    number       = {},
    pages        = {1--10},
    doi          = {10.1109/InPar.2012.6339595},
    keywords     = {
        Graphics processing unit;Abstracts;Programming;Nickel;Tiles;Benchmark
        testing;Auto-tuning;GPU;CUDA;OpenCL;Optimization;Belief Propagation
    }
}
```
