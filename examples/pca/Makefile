INCLUDES=-I/usr/local/include/eigen3
C_VERSION=c++17
## OPT_FLAGS=-g
OPT_FLAGS=-fopenmp
EXE_FLAGS=-std=$(C_VERSION) $(OPT_FLAGS) $(INCLUDES)
OBJ_FLAGS=$(EXE_FLAGS) -c

all: pca
clean:
	rm -rf *.o pca

pca: pca.cxx
	g++ $(EXE_FLAGS) -o $@ $^

## eof - $RCSfile$
