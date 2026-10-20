#ifndef NN
#define NN
typedef struct Network{
    double weight_input1_hidden1;
    double weight_input2_hiddem1;
    double weight_input1_hidden2;
    double weight_input2_hidden2;

    double weight_hidden1_output;
    double weight_hidden2_output;

    double biashidden1;
    double biashidden2;
    double biasoutput;
} Network;

void init_network(Network *nn);

#endif
