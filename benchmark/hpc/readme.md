# HPC benchmarks

## PolyBench/GPU (poly)

[PolyBench/GPU](https://www.cs.colostate.edu/~pouchet/software/polybench/GPU/index.html) (a.k.a. [sgrauerg/polybenchGpu](https://github.com/sgrauerg/polybenchGpu)) is a collection of [the Polyhedral Benchmark suite](https://www.cs.colostate.edu/~pouchet/software/polybench/) codes (as well as convolution) implemented for processing on the GPU using CUDA, OpenCL, and HMPP (pragma-based compiler).

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

## Rodinia (rodinia)

[Rodiana](https://rodinia.cs.virginia.edu/) (a.k.a. [yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia)) is a benchmark suite for heterogeneous computing, including applications and kernels which target multi-core CPU and GPU platforms.

```bibtex
@inproceedings{rodinia,
    title        = {Rodinia: A benchmark suite for heterogeneous computing},
    author       = {
        Che, Shuai and Boyer, Michael and Meng, Jiayuan and Tarjan, David and
        Sheaffer, Jeremy W. and Lee, Sang-Ha and Skadron, Kevin
    },
    year         = 2009,
    booktitle    = {
        Proceedings of the 2009 IEEE International Symposium on Workload
        Characterization (IISWC)
    },
    publisher    = {IEEE Computer Society},
    address      = {USA},
    series       = {IISWC '09},
    pages        = {44–54},
    doi          = {10.1109/IISWC.2009.5306797},
    isbn         = 9781424451562,
    url          = {https://doi.org/10.1109/IISWC.2009.5306797},
    abstract     = {
        This paper presents and characterizes Rodinia, a benchmark suite for
        heterogeneous computing. To help architects study emerging platforms
        such as GPUs (Graphics Processing Units), Rodinia includes applications
        and kernels which target multi-core CPU and GPU platforms. The choice
        of applications is inspired by Berkeley's dwarf taxonomy. Our
        characterization shows that the Rodinia benchmarks cover a wide range
        of parallel communication patterns, synchronization techniques and
        power consumption, and has led to some important architectural insight,
        such as the growing importance of memory-bandwidth limitations and the
        consequent importance of data layout.
    },
    numpages     = 11
}
```

## SHOC (shoc)

