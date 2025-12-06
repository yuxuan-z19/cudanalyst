#include "XSbench_header.cuh"

__device__ long grid_search_custom(long n, double quarry,
                                   double* __restrict__ A) {
    long lowerLimit = 0;
    long upperLimit = n - 1;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while (length > 1) {
        examinationPoint = lowerLimit + (length / 2);

        if (A[examinationPoint] > quarry)
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
}

__host__ __device__ long grid_search_nuclide_custom(long n, double quarry,
                                                    NuclideGridPoint* A,
                                                    long low, long high) {
    long lowerLimit = low;
    long upperLimit = high;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while (length > 1) {
        examinationPoint = lowerLimit + (length / 2);

        if (A[examinationPoint].energy > quarry)
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
}

__device__ void calculate_micro_xs_custom(
    double p_energy, int nuc, long n_isotopes, long n_gridpoints,
    double* __restrict__ egrid, int* __restrict__ index_data,
    NuclideGridPoint* __restrict__ nuclide_grids, long idx,
    double* __restrict__ xs_vector, int grid_type, int hash_bins) {
    double f;
    NuclideGridPoint *low, *high;

    if (grid_type == NUCLIDE) {
        idx = grid_search_nuclide_custom(n_gridpoints, p_energy,
                                         &nuclide_grids[nuc * n_gridpoints], 0,
                                         n_gridpoints - 1);

        if (idx == n_gridpoints - 1)
            low = &nuclide_grids[nuc * n_gridpoints + idx - 1];
        else
            low = &nuclide_grids[nuc * n_gridpoints + idx];
    } else if (grid_type == UNIONIZED)

    {
        if (index_data[idx * n_isotopes + nuc] == n_gridpoints - 1)
            low = &nuclide_grids[nuc * n_gridpoints +
                                 index_data[idx * n_isotopes + nuc] - 1];
        else
            low = &nuclide_grids[nuc * n_gridpoints +
                                 index_data[idx * n_isotopes + nuc]];
    } else {
        int u_low = index_data[idx * n_isotopes + nuc];

        int u_high;
        if (idx == hash_bins - 1)
            u_high = n_gridpoints - 1;
        else
            u_high = index_data[(idx + 1) * n_isotopes + nuc] + 1;

        double e_low = nuclide_grids[nuc * n_gridpoints + u_low].energy;
        double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
        int lower;
        if (p_energy <= e_low)
            lower = 0;
        else if (p_energy >= e_high)
            lower = n_gridpoints - 1;
        else
            lower = grid_search_nuclide_custom(
                n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints],
                u_low, u_high);

        if (lower == n_gridpoints - 1)
            low = &nuclide_grids[nuc * n_gridpoints + lower - 1];
        else
            low = &nuclide_grids[nuc * n_gridpoints + lower];
    }

    high = low + 1;

    f = (high->energy - p_energy) / (high->energy - low->energy);

    xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);

    xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);

    xs_vector[2] =
        high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);

    xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);

    xs_vector[4] =
        high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

__device__ void calculate_macro_xs_custom(
    double p_energy, int mat, long n_isotopes, long n_gridpoints,
    int* __restrict__ num_nucs, double* __restrict__ concs,
    double* __restrict__ egrid, int* __restrict__ index_data,
    NuclideGridPoint* __restrict__ nuclide_grids, int* __restrict__ mats,
    double* __restrict__ macro_xs_vector, int grid_type, int hash_bins,
    int max_num_nucs) {
    int p_nuc;
    long idx = -1;
    double conc;

    for (int k = 0; k < 5; k++) macro_xs_vector[k] = 0;

    if (grid_type == UNIONIZED)
        idx = grid_search_custom(n_isotopes * n_gridpoints, p_energy, egrid);
    else if (grid_type == HASH) {
        double du = 1.0 / hash_bins;
        idx = p_energy / du;
    }

    for (int j = 0; j < num_nucs[mat]; j++) {
        double xs_vector[5];
        p_nuc = mats[mat * max_num_nucs + j];
        conc = concs[mat * max_num_nucs + j];
        calculate_micro_xs_custom(p_energy, p_nuc, n_isotopes, n_gridpoints,
                                  egrid, index_data, nuclide_grids, idx,
                                  xs_vector, grid_type, hash_bins);
        for (int k = 0; k < 5; k++) macro_xs_vector[k] += xs_vector[k] * conc;
    }
}

__global__ void xs_lookup_kernel_optimization_custom(Inputs in,
                                                     SimulationData GSD, int m,
                                                     int n_lookups,
                                                     int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_lookups) return;

    i += offset;

    int mat = GSD.mat_samples[i];
    if (mat != m) return;

    double macro_xs_vector[5] = {0};

    calculate_macro_xs_custom(GSD.p_energy_samples[i], mat, in.n_isotopes,
                              in.n_gridpoints, GSD.num_nucs, GSD.concs,

                              GSD.unionized_energy_array, GSD.index_grid,

                              GSD.nuclide_grid,

                              GSD.mats,

                              macro_xs_vector,

                              in.grid_type, in.hash_bins, GSD.max_num_nucs);

    double max = -1.0;
    int max_idx = 0;
    for (int j = 0; j < 5; j++) {
        if (macro_xs_vector[j] > max) {
            max = macro_xs_vector[j];
            max_idx = j;
        }
    }
    GSD.verification[i] = max_idx + 1;
}

unsigned long long run_event_based_simulation_optimization_custom(
    Inputs in, SimulationData GSD, int _) {
    const char* optimization_name = "Custom Optimization";

    size_t sz;
    size_t total_sz = 0;

    sz = in.lookups * sizeof(double);
    gpuErrchk(cudaMalloc((void**)&GSD.p_energy_samples, sz));
    total_sz += sz;
    GSD.length_p_energy_samples = in.lookups;

    sz = in.lookups * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&GSD.mat_samples, sz));
    total_sz += sz;
    GSD.length_mat_samples = in.lookups;

    int nthreads = 32;
    int nblocks = ceil((double)in.lookups / 32.0);

    sampling_kernel<<<nblocks, nthreads>>>(in, GSD);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int n_lookups_per_material[12];
    for (int m = 0; m < 12; m++)
        n_lookups_per_material[m] = thrust::count(
            thrust::device, GSD.mat_samples, GSD.mat_samples + in.lookups, m);

    thrust::sort_by_key(thrust::device, GSD.mat_samples,
                        GSD.mat_samples + in.lookups, GSD.p_energy_samples);

    int offset = 0;
    for (int m = 0; m < 12; m++) {
        thrust::sort_by_key(
            thrust::device, GSD.p_energy_samples + offset,
            GSD.p_energy_samples + offset + n_lookups_per_material[m],
            GSD.mat_samples + offset);
        offset += n_lookups_per_material[m];
    }

    offset = 0;
    for (int m = 0; m < 12; m++) {
        nthreads = 32;
        nblocks = ceil((double)n_lookups_per_material[m] / (double)nthreads);
        xs_lookup_kernel_optimization_custom<<<nblocks, nthreads>>>(
            in, GSD, m, n_lookups_per_material[m], offset);
        offset += n_lookups_per_material[m];
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    unsigned long verification_scalar = thrust::reduce(
        thrust::device, GSD.verification, GSD.verification + in.lookups, 0);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return verification_scalar;
}