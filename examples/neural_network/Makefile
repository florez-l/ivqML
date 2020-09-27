
all: test binary_classifier
clean:
	rm -rf *.o test binary_classifier

test: ActivationFunctions.o Layer.o NeuralNetwork.o test.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -o $@ $^

binary_classifier: CSVReader.o ActivationFunctions.o Layer.o NeuralNetwork.o binary_classifier.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -o $@ $^

%.o: %.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -c -o $@ $^

## eof - $RCSfile$
