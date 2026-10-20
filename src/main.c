#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"
void infer(Network *nn);
void train(Network *nn);
int main(){
    Network nn;
    srand((unsigned int)time(NULL));
    init_network(&nn);
    if (load_weights(&nn, "weights.csv")) {
        printf("Loaded saved weights from weights.csv\n\n");
    }
    while(1){
        printf("XOR NEURAL NETWORK\nCHOOSE AN OPTION\n1 - TRAINING\n2 - INFERENCE\n3 - QUIT\n");
        int option;
        scanf("%d", &option);
        switch (option) {
            case 1: train(&nn);
                    break;
            case 2: infer(&nn);
                    break;
            case 3: return 0;
                    break;
            default:printf("Invalid Input, Try again\n\n");
                    continue;
        }
    }
    return 0;
}

void train(Network *nn){
    int epochs;
    printf("Enter number of epochs: ");
    if (scanf("%d", &epochs) != 1 || epochs <= 0) {
        printf("Invalid input for epochs\n\n");
        return;
    }
    const double learning_rate = 0.1;
    train_network(nn, epochs, learning_rate);
    if (save_weights(nn, "weights.csv")) {
        printf("Weights saved to weights.csv successfully\n\n");
    } else {
        printf("Failed to save weights to weights.csv\n\n");
    }
}

void infer(Network *nn){
    double input1, input2;
    printf("Enter 2 binary inputs to calculate XOR for them\n");
    scanf("%lf %lf", &input1, &input2);
    ForwardResult result = forward(nn, input1, input2);
    int prediction = result.output >= 0.5? 1:0;
    printf("Output : %.4f\nPrediction : %d\n", result.output, prediction);
}

