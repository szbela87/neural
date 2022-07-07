FC = nvcc
FFLAGS = -Xcompiler -fopenmp -O3 -use_fast_math 

all: neural

neural: kernels.o
	nvcc -o neural -lcuda -lcudart main.cu kernels.o $(FFLAGS)

kernels.o:
	nvcc -c kernels.cu 

clean:
	rm -f neural
	rm -f *.o

