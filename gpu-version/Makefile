FC = nvcc
FFLAGS = -O3 -lm  

all: neural

neural: kernels.o
	nvcc -o neural -lcuda -lcudart main.cu kernels.o $(FFLAGS)

kernels.o:
	nvcc -c kernels.cu 

clean:
	rm -f neural
	rm -f *.o

