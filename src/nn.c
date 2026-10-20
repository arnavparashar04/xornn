#include "nn.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
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

ForwardResult forward(Network *n, double input1, double input2){
    double h1ws = input1*n->weight_input1_hidden1 + input2*n->weight_input2_hidden1 + n->biashidden1; 
    double h2ws = input1*n->weight_input1_hidden2 + input2*n->weight_input2_hidden2 + n->biashidden2;
    double h1activation = sigmoid(h1ws);
    double h2activation = sigmoid(h2ws);
    double ows = n->weight_hidden1_output*h1activation + n->weight_hidden2_output*h2activation + n->biasoutput;
    
    double output = sigmoid(ows);
    ForwardResult result = {.h1 = h1activation, .h2 = h2activation, .output = output};
    return result;
}

void train_network(Network *nn, int epochs, double learning_rate){
    const double inputds[4][2] = {{0,0}, {0,1}, {1,0}, {1, 1}};
    const double targetds[4] = {0,1,1,0};

    for(int epoch = 1; epoch<=epochs; epoch++){
        double totalLoss = 0.0;
        for(int i = 0; i<4; i++){
            double x1 = inputds[i][0];
            double x2 = inputds[i][1];
            double target = targetds[i];

            ForwardResult result = forward(nn, x1, x2);
            double h1 = result.h1;
            double h2 = result.h2;
            double output = result.output;

            totalLoss += loss(target, output);

            double delOutput = (output - target) *output*(1.0-output);
            double delh1 = delOutput * nn->weight_hidden1_output *h1*(1.0 - h1);
            double delh2 = delOutput * nn->weight_hidden2_output *h2*(1.0 - h2);
            
            nn->weight_hidden1_output -= delOutput*h1*learning_rate;
            nn->weight_hidden2_output -= delOutput*h2*learning_rate;
            nn->biasoutput -= learning_rate*delOutput;

            nn->weight_input1_hidden1 -= learning_rate*delh1*x1;
            nn->weight_input2_hidden1 -= learning_rate*delh1*x2;
            nn->biashidden1 -= learning_rate*delh1;
            
            nn->weight_input1_hidden2 -= learning_rate*delh2*x1;
            nn->weight_input2_hidden2 -= learning_rate*delh2*x2;
            nn->biashidden2 -= learning_rate*delh2;
        }
        if ((epoch) % 1000 == 0) {
            printf("Epoch %d | Average loss: %.6f\n",epoch + 1, totalLoss / 4.0);
        }
    }
}

int save_weights(const Network *nn, const char *filename){
    FILE *file = fopen(filename, "w");
    if(file == NULL){
        return 0;
    }
    fprintf(file, "weight_input1_hidden1,weight_input2_hidden1,weight_input1_hidden2,weight_input2_hidden2,weight_hidden1_output,weight_hidden2_output,biashidden1,biashidden2,biasoutput\n");
    fprintf(file, "%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
            nn->weight_input1_hidden1,
            nn->weight_input2_hidden1,
            nn->weight_input1_hidden2,
            nn->weight_input2_hidden2,
            nn->weight_hidden1_output,
            nn->weight_hidden2_output,
            nn->biashidden1,
            nn->biashidden2,
            nn->biasoutput);
    fclose(file);
    return 1;
}

int load_weights(Network *nn, const char *filename){
    FILE *file = fopen(filename, "r");
    if(file == NULL){
        return 0;
    }
    char line[512];
    if(fgets(line, sizeof(line), file) == NULL){
        fclose(file);
        return 0;
    }
    int count = fscanf(file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                       &nn->weight_input1_hidden1,
                       &nn->weight_input2_hidden1,
                       &nn->weight_input1_hidden2,
                       &nn->weight_input2_hidden2,
                       &nn->weight_hidden1_output,
                       &nn->weight_hidden2_output,
                       &nn->biashidden1,
                       &nn->biashidden2,
                       &nn->biasoutput);
    if(count == 9){
        fclose(file);
        return 1;
    }
    count = sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &nn->weight_input1_hidden1,
                   &nn->weight_input2_hidden1,
                   &nn->weight_input1_hidden2,
                   &nn->weight_input2_hidden2,
                   &nn->weight_hidden1_output,
                   &nn->weight_hidden2_output,
                   &nn->biashidden1,
                   &nn->biashidden2,
                   &nn->biasoutput);
    fclose(file);
    return count == 9;
}

