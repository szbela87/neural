FC = gcc
FFLAGS = -O3 -lm -fopenmp

all: neural
neural: main.c
	$(FC) -o neural main.c $(FFLAGS)

clean:
	rm -f neural
	rm -f *.o

.PHONY: all clean

