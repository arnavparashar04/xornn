#ifndef NN
#define NN
typedef struct Network{
    double weight_input1_hidden1;
    double weight_input2_hidden1;
    double weight_input1_hidden2;
    double weight_input2_hidden2;

    double weight_hidden1_output;
    double weight_hidden2_output;

    double biashidden1;
    double biashidden2;
    double biasoutput;
} Network;

typedef struct ForwardResult{
    double h1;
    double h2;
    double output;
} ForwardResult;

void init_network(Network *nn);
ForwardResult forward(Network *n, double input1, double input2);
void train_network(Network *nn, int epochs, double learning_rate);
int save_weights(const Network *nn, const char *filename);
int load_weights(Network *nn, const char *filename);

#endif
