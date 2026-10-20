#include "nn.h"
#include <stdlib.h>
#include <math.h>
void init_network(Network *nn){
    nn->weight_input1_hidden1 = (double)rand() / RAND_MAX * 2.0 - 1.0; //shifr range to -1 to 1
    nn->weight_input2_hidden1 = (double)rand() / RAND_MAX * 2.0 - 1.0;
    nn->weight_input1_hidden2 = (double)rand() / RAND_MAX * 2.0 - 1.0;
    nn->weight_input2_hidden2 = (double)rand() / RAND_MAX * 2.0 - 1.0;
 
    nn->weight_hidden1_output = (double)rand() / RAND_MAX * 2.0 - 1.0;
    nn->weight_hidden2_output = (double)rand() / RAND_MAX * 2.0 - 1.0;

    nn->biashidden1 = 0.0;
    nn->biashidden2 = 0.0;
    nn->biasoutput = 0.0;
}


// mathematical helpers here

static double sigmoid(double x){
    return 1.0/(1.0 + exp(-x));
}

static double sigmoid_derivative(double x){
    return x*(1.0 - x);
}

static double loss(double target, double prediction){
    double error = target - prediction;
    return 0.5 * error*error;
}
