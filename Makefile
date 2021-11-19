CC = g++

CCFLAGSBASE = -std=c++20 -march=native -O3 -fno-trapping-math -fabi-version=0 -funroll-loops -ffast-math -fargument-noalias
SSE2FLAGS = -ftree-vectorize -msse2 -fopt-info-vec
OPENMP = -fopenmp
LFLAGS_OMP = -lm -lpthread

all: stencil

stencil: stencil.cc Makefile
	$(CC) $(CCFLAGSBASE) $(SSE2FLAGS) $(OPENMP) $(LFLAGS_OMP) -o $@ $<

run:
	./stencil

clean:
	rm -rf stencil
