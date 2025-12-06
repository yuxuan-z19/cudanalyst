all:
	nvcc -O3 ${CUFILES} -o ${EXECUTABLE} -I "../../common" ${LDFLAGS}
clean:
	rm -f *~ *.exe