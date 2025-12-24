SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=LU

include ../config/make.def

CU_SRCS := $(filter-out %.init.cu,$(wildcard *.cu))
CU_OBJS := $(CU_SRCS:.cu=.o)

OBJS = $(CU_OBJS) \
       ${COMMON}/c_print_results.o \
       ${COMMON}/c_${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

%.o: %.cu ${BENCHMARK}.cuh npbparams.hpp
	${CCOMPILE} $<

clean:
	- /bin/rm -f npbparams.hpp
	- /bin/rm -f *.o *~
