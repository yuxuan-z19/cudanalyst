#if !defined(DOITGEN_CUH)
#define DOITGEN_CUH

/* Problem size. */
#define NR 128
#define NQ 128
#define NP 128

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#endif  // DOITGEN_CUH
