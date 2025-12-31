#include <nvtx3/nvToolsExt.h>

#include "XSbench_header.cuh"

using kernel_func_t = unsigned long long (*)(Inputs, SimulationData, int);
kernel_func_t kernel_table[] = {run_event_based_simulation_baseline,
                                run_event_based_simulation_optimization_1,
                                run_event_based_simulation_optimization_2,
                                run_event_based_simulation_optimization_3,
                                run_event_based_simulation_optimization_4,
                                run_event_based_simulation_optimization_5,
                                run_event_based_simulation_optimization_6,
                                run_event_based_simulation_optimization_custom};
static constexpr int NUM_KERNELS = sizeof(kernel_table) / sizeof(kernel_func_t);

int main(int argc, char* argv[]) {
    // =====================================================================
    // Initialization & Command Line Read-In
    // =====================================================================
    int version = 20;
    // int mype = 0;
    int mype = 1;  // * disable printf
    double omp_start, omp_end, runtime;
    int nprocs = 1;
    unsigned long long verification;

    // Process CLI Fields -- store in "Inputs" structure
    Inputs in = read_CLI(argc, argv);

    // Print-out of Input Summary
    if (mype == 0) print_inputs(in, nprocs, version);

    // =====================================================================
    // Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
    // This is not reflective of a real Monte Carlo simulation workload,
    // therefore, do not profile this region!
    // =====================================================================

    SimulationData SD;

    // If read from file mode is selected, skip initialization and load
    // all simulation data structures from file instead
    if (in.binary_mode == READ)
        SD = binary_read(in);
    else
        SD = grid_init_do_not_profile(in, mype);

    // If writing from file mode is selected, write all simulation data
    // structures to file
    if (in.binary_mode == WRITE && mype == 0) binary_write(in, SD);

    // Move data to GPU
    SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

    // =====================================================================
    // Cross Section (XS) Parallel Lookup Simulation
    // This is the section that should be profiled, as it reflects a
    // realistic continuous energy Monte Carlo macroscopic cross section
    // lookup kernel.
    // =====================================================================
    if (mype == 0) {
        printf("\n");
        border_print();
        center_print("SIMULATION", 79);
        border_print();
    }

    // Run simulation
    if (in.simulation_method == EVENT_BASED) {
        int id = in.kernel_id;
        if (id < 0 || id >= NUM_KERNELS) {
            printf("Error: No kernel ID %d found!\n", in.kernel_id);
            exit(1);
        }

        // Warmup
        for (int i = 0; i < WARMUP; i++) kernel_table[id](in, GSD, mype);

        // Start Simulation Timer
        omp_start = get_time();
        for (int i = 0; i < ITER; i++)
            verification = kernel_table[id](in, GSD, mype);
        // End Simulation Timer
        omp_end = get_time();
        runtime = (omp_end - omp_start) / ITER;

        // Single NCU Profiling
        nvtxRangePushA("cugedit");
        verification = kernel_table[id](in, GSD, mype);
        cudaDeviceSynchronize();
        nvtxRangePop();
    } else {
        printf(
            "History-based simulation not implemented in CUDA code. "
            "Instead,\nuse the event-based method with \"-m event\" "
            "argument.\n");
        exit(1);
    }

    if (mype == 0) {
        printf("\n");
        printf("Simulation complete.\n");
    }

    // Release device memory
    release_device_memory(GSD);

    // Final Hash Step
    verification = verification % 999983;

    // Print / Save Results and Exit
    int is_invalid_result = print_results(in, 0, runtime, nprocs, verification);

    return is_invalid_result;
}
