#include "include/mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

//3 hidden layers with 10 neurons, 784 inputs, 10 outputs (1 -> 2 -> 3 -> output)

double output_Layer[10] = { 0 };
double Layer_1[10] = { 0 };
double Layer_2[10] = { 0 };
double Layer_3[10] = { 0 };

double sum_Output[10] = { 0 };
double sum_Layer_1[10] = { 0 };
double sum_Layer_2[10] = { 0 };
double sum_Layer_3[10] = { 0 };

double weights_Layer_1[784][10] = { 0 };
double weights_Layer_2[10][10] = { 0 };
double weights_Layer_3[10][10] = { 0 };
double weights_Output[10][10] = { 0 };

double bias_Layer_1[10] = { 0 };
double bias_Layer_2[10] = { 0 };
double bias_Layer_3[10] = { 0 };
double bias_Output[10] = { 0 };

double d_weights_Layer_1[784][10] = { 0 };
double d_weights_Layer_2[10][10] = { 0 };
double d_weights_Layer_3[10][10] = { 0 };
double d_weights_Output[10][10] = { 0 };

double d_bias_Layer_1[10] = { 0 };
double d_bias_Layer_2[10] = { 0 };
double d_bias_Layer_3[10] = { 0 };
double d_bias_Output[10] = { 0 };

double sum_d_weights_Layer_1[784][10] = { 0 }; //todo: move back into minibatch?
double sum_d_weights_Layer_2[10][10] = { 0 };
double sum_d_weights_Layer_3[10][10] = { 0 };
double sum_d_weights_Output[10][10] = { 0 };

double sum_d_bias_Layer_1[10] = { 0 };
double sum_d_bias_Layer_2[10] = { 0 };
double sum_d_bias_Layer_3[10] = { 0 };
double sum_d_bias_Output[10] = { 0 };

//function definitions

int initialise_Weight_Matrices(void);
int initialise_Bias_Matrices(void);

double sigmoid(double exponent);
int forward_Prop(int imageNum, bool testFlag);
int back_prop(int imageNum, bool testFlag);

int minibatch(int trainExNum, double learningRate);
int update_Weight_Matrices(double learningRate);
int update_Bias_Matrices(double learningRate);

double test_Network(int imageNum);

int main(void)
{
    srand((unsigned)time(NULL));
    // call to store mnist in array
    load_mnist();
    //initialise weight matrices (-0.1 to 0.1) todo: make range parameters
    initialise_Weight_Matrices();
    //initialise bias matrices (-0.1 to 0.1)
    initialise_Bias_Matrices();
    
    forward_Prop(0, false);
    back_prop(0, false);
    int i;
    for (i = 0; i < 10; i++)
    {
        printf("%f \n", output_Layer[i]);
    }

    for (i = 0; i < 50; i++)
    {
        minibatch(100, 0.1);
    }
    

    printf("Epoch Accuracy = %f \n", test_Network(100));
    
    for (i = 0; i < 784; i++)
    {
        if ((i % 28) == 0)
        {
            printf("\n");
        }
        printf("%.1f ", train_image[1000][i]);
    }
    printf("%d\n", train_label[1000]);

    // print first label in test dataset
    //printf("label: %d\n", test_label[0]);

    // save image of first data in test dataset as .pgm file
    //save_mnist_pgm(test_image, 0);

    // show all pixels and labels in test dataset
    //print_mnist_pixel(test_image, NUM_TEST);
    //print_mnist_label(test_label, NUM_TEST);

    return 0;
}

int initialise_Weight_Matrices(void)
{
    int x, y;
    for (x = 0; x < 784; x++) //layer 1
    {
        for (y = 0; y < 10; y++)
        {
            weights_Layer_1[x][y] = ((double)rand() / RAND_MAX * 0.2 - 0.1); //random values between -0.1 and 0.1
        }
    }
    for (x = 0; x < 10; x++)//layer 2,3,output
    {
        for (y = 0; y < 10; y++)
        {
            weights_Layer_2[x][y] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
            weights_Layer_3[x][y] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
            weights_Output[x][y] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        }
    }
    return 0;
}

int initialise_Bias_Matrices(void)
{
    int x;
    for (x = 0; x < 10; x++)
    {
        bias_Layer_1[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        bias_Layer_2[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        bias_Layer_3[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        bias_Output[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
    }
    return 0;
}

double sigmoid(double exponent)
{
    return 1 / (1 + exp(-1 * exponent));
}

int forward_Prop(int imageNum, bool testFlag) //sigmoid on output/tanh  in hidden
{
    double sum;
    int x, y; //y = neuron index, x = weight index
    
    //layer 1
    for (y = 0; y < 10; y++)
    {
        sum = 0.0;
        for (x = 0; x < 784; x++)
        {
            if (testFlag)
            {
                sum = sum + test_image[imageNum][x] * weights_Layer_1[x][y];
            }
            else
            {
                sum = sum + train_image[imageNum][x] * weights_Layer_1[x][y];
            }

            //sum = sum + ((double)!testFlag)*(train_image[imageNum][x] * weights_Layer_1[x][y]) + ((double)testFlag) * (test_image[imageNum][x] * weights_Layer_1[x][y]);
            //todo: redo branchless, cast bool to double?
        }
        sum = sum + bias_Layer_1[y];
        if (!testFlag) //todo: cache always?
        {
            sum_Layer_1[y] = sum;
        }
        Layer_1[y] = tanh(sum);
    }

    //layer 2
    for (y = 0; y < 10; y++)
    {
        sum = 0.0;
        for (x = 0; x < 10; x++)
        {
            sum = sum + Layer_1[y] * weights_Layer_2[x][y];
        }
        sum = sum + bias_Layer_2[y];
        if (!testFlag)
        {
            sum_Layer_2[y] = sum;
        }
        Layer_2[y] = tanh(sum);
    }

    //layer 3
    for (y = 0; y < 10; y++)
    {
        sum = 0.0;
        for (x = 0; x < 10; x++)
        {
            sum = sum + Layer_2[y] * weights_Layer_3[x][y];
        }
        sum = sum + bias_Layer_3[y];
        if (!testFlag)
        {
            sum_Layer_3[y] = sum;
        }
        Layer_3[y] = tanh(sum);
    }

    //output layer
    for (y = 0; y < 10; y++)
    {
        sum = 0.0;
        for (x = 0; x < 10; x++)
        {
            sum = sum + Layer_3[y] * weights_Output[x][y];
        }
        sum = sum + bias_Output[y];
        if (!testFlag)
        {
            sum_Output[y] = sum;
        }
        output_Layer[y] = sigmoid(sum);
    }

    return 0;
}

int back_prop(int imageNum, bool testFlag) //cross entropy loss: -label*log(est) - (1-label)*log(1-est) dir: -(label/a)+(1-label)(1-a)
{   //todo: remove bias at output? lyk nie so nie
    //todo: check of cache nodig is
    double d_output_Layer[10] = { 0 }; //da
    double d_Layer_1[10] = { 0 };
    double d_Layer_2[10] = { 0 };
    double d_Layer_3[10] = { 0 };

    double d_sum_Output[10] = { 0 }; //dz
    double d_sum_Layer_1[10] = { 0 };
    double d_sum_Layer_2[10] = { 0 };
    double d_sum_Layer_3[10] = { 0 };

    double sum;
    int i, x, y;
    for (i = 0; i < 10; i++) //todo: branchless? sub label from index? //output layer
    {
        if (i == (testFlag * test_label[imageNum] + !testFlag * train_label[imageNum])) //todo: time branchless //da
        {
            d_output_Layer[i] = -(1 / output_Layer[i]);
        }
        else 
        {
            d_output_Layer[i] = (1 / (1 - output_Layer[i]));
        }
        d_sum_Output[i] = d_output_Layer[i] * output_Layer[i] * (1 - output_Layer[i]); //dz
    }

    for (y = 0; y < 10; y++) 
    {
        for (x = 0; x < 10; x++)
        {
            d_weights_Output[x][y] = Layer_3[x] * d_sum_Output[y]; //dw
        }
        d_bias_Output[y] = d_sum_Output[y]; //db
    }


    ///////////////////////////// end output

    for (x = 0; x < 10; x++) //dz(L) -> da(L-1)
    {
        sum = 0;
        for (y = 0; y < 10; y++)
        {
            sum = sum + weights_Output[x][y] * d_sum_Output[y];
        }
        d_Layer_3[x] = sum;
    }

    for (i = 0; i < 10; i++) //tanh dir: 1-a^2
    {
        d_sum_Layer_3[i] = d_Layer_3[i] * (1 - (Layer_3[i] * Layer_3[i])); //dz
    }
    
    for (y = 0; y < 10; y++)
    {
        for (x = 0; x < 10; x++)
        {
            d_weights_Layer_3[x][y] = Layer_2[x] * d_sum_Layer_3[y]; //dw
        }
        d_bias_Layer_3[y] = d_sum_Layer_3[y]; //db
    }
    //////////////////////////// end layer 3

    for (x = 0; x < 10; x++) //dz(L) -> da(L-1)
    {
        sum = 0;
        for (y = 0; y < 10; y++)
        {
            sum = sum + weights_Layer_3[x][y] * d_sum_Layer_3[y];
        }
        d_Layer_2[x] = sum;
    }

    for (i = 0; i < 10; i++) //tanh dir: 1-a^2
    {
        d_sum_Layer_2[i] = d_Layer_2[i] * (1 - (Layer_2[i] * Layer_2[i])); //dz
    }

    for (y = 0; y < 10; y++)
    {
        for (x = 0; x < 10; x++)
        {
            d_weights_Layer_2[x][y] = Layer_1[x] * d_sum_Layer_2[y]; //dw
        }
        d_bias_Layer_2[y] = d_sum_Layer_2[y]; //db
    }

    //////////////////////////////// end layer 2
    
    for (x = 0; x < 10; x++) //dz(L) -> da(L-1)
    {
        sum = 0;
        for (y = 0; y < 10; y++)
        {
            sum = sum + weights_Layer_2[x][y] * d_sum_Layer_2[y];
        }
        d_Layer_1[x] = sum;
    }

    for (i = 0; i < 10; i++) //tanh dir: 1-a^2
    {
        d_sum_Layer_1[i] = d_Layer_1[i] * (1 - (Layer_1[i] * Layer_1[i])); //dz
    }

    double diff = 0;
    for (y = 0; y < 10; y++)
    {
        for (x = 0; x < 784; x++)
        {
            //d_weights_Layer_1[x][y] = (testFlag*test_image[imageNum][x] + !testFlag*train_image[imageNum][x]) * d_sum_Layer_1[y];  todo: redo branchless

            if (testFlag) //dw
            {
                d_weights_Layer_1[x][y] = test_image[imageNum][x] * d_sum_Layer_1[y];
            }
            else
            {
                d_weights_Layer_1[x][y] = train_image[imageNum][x] * d_sum_Layer_1[y];
                if (train_image[imageNum][x] != 0)
                {
                    diff = train_image[imageNum][x] * d_sum_Layer_1[y];
                }
            }
        }
        d_bias_Layer_1[y] = d_sum_Layer_1[y]; //db
    }


    return 0;
}

int minibatch(int trainExNum, double learningRate) 
{
    int randEx, i, x, y;
    bool finalExFlag = false;

    
    for (i = 0; i < trainExNum; i++)
    {
        randEx = rand() % 60000;

        forward_Prop(randEx, false);
        back_prop(randEx, false);
        if (trainExNum - i == 1)
        {
            finalExFlag = true;
        }
        for (y = 0; y < 10; y++) //avg bias grad
        {
            sum_d_bias_Layer_1[y] = sum_d_bias_Layer_1[y] + d_bias_Layer_1[y];
            sum_d_bias_Layer_2[y] = sum_d_bias_Layer_2[y] + d_bias_Layer_2[y];
            sum_d_bias_Layer_3[y] = sum_d_bias_Layer_3[y] + d_bias_Layer_3[y];
            sum_d_bias_Output[y] = sum_d_bias_Output[y] + d_bias_Output[y];
            if (finalExFlag)
            {
                d_bias_Layer_1[y] = sum_d_bias_Layer_1[y] / (double)trainExNum;
                d_bias_Layer_2[y] = sum_d_bias_Layer_2[y] / (double)trainExNum;
                d_bias_Layer_3[y] = sum_d_bias_Layer_3[y] / (double)trainExNum;
                d_bias_Output[y] = sum_d_bias_Output[y] / (double)trainExNum;

                sum_d_bias_Layer_1[y] = 0;
                sum_d_bias_Layer_2[y] = 0;
                sum_d_bias_Layer_3[y] = 0;
                sum_d_bias_Output[y] = 0;
            }
        }

        for (y = 0; y < 10; y++) //avg weight grad
        {
            for (x = 0; x < 784; x++)
            {
                sum_d_weights_Layer_1[x][y] = sum_d_weights_Layer_1[x][y] + d_weights_Layer_1[x][y];
                if (finalExFlag)
                {
                    d_weights_Layer_1[x][y] = sum_d_weights_Layer_1[x][y] / (double)trainExNum;

                    sum_d_weights_Layer_1[x][y] = 0;
                }
            }
            for (x = 0; x < 10; x++)
            {
                sum_d_weights_Layer_2[x][y] = sum_d_weights_Layer_2[x][y] + d_weights_Layer_2[x][y];
                sum_d_weights_Layer_3[x][y] = sum_d_weights_Layer_3[x][y] + d_weights_Layer_3[x][y];
                sum_d_weights_Output[x][y] = sum_d_weights_Output[x][y] + d_weights_Output[x][y];
                if (finalExFlag)
                {
                    d_weights_Layer_2[x][y] = sum_d_weights_Layer_2[x][y] / (double)trainExNum;
                    d_weights_Layer_3[x][y] = sum_d_weights_Layer_3[x][y] / (double)trainExNum;
                    d_weights_Output[x][y] = sum_d_weights_Output[x][y] / (double)trainExNum;

                    sum_d_weights_Layer_2[x][y] = 0;
                    sum_d_weights_Layer_3[x][y] = 0;
                    sum_d_weights_Output[x][y] = 0;
                }
            }
        }
    }
    
    update_Bias_Matrices(learningRate);
    update_Weight_Matrices(learningRate);
    return 0;
}

int update_Weight_Matrices(double learningRate)
{
    int x, y;

    for (y = 0; y < 10; y++)
    {
        for (x = 0; x < 784; x++)
        {
            weights_Layer_1[x][y] = weights_Layer_1[x][y] - learningRate * d_weights_Layer_1[x][y];
        }
        for (x = 0; x < 10; x++)
        {
            weights_Layer_2[x][y] = weights_Layer_2[x][y] - learningRate * d_weights_Layer_2[x][y];
            weights_Layer_3[x][y] = weights_Layer_3[x][y] - learningRate * d_weights_Layer_3[x][y];
            weights_Output[x][y] = weights_Output[x][y] - learningRate * d_weights_Output[x][y];
        }
    }
    return 0;
}

int update_Bias_Matrices(double learningRate)
{
    int y;

    for (y = 0; y < 10; y++)
    {
        bias_Layer_1[y] = bias_Layer_1[y] - learningRate * d_bias_Layer_1[y];
        bias_Layer_2[y] = bias_Layer_2[y] - learningRate * d_bias_Layer_2[y];
        bias_Layer_3[y] = bias_Layer_3[y] - learningRate * d_bias_Layer_3[y];
        bias_Output[y] = bias_Output[y] - learningRate * d_bias_Output[y];
    }
    return 0;
}

double test_Network(int testExNum)
{
    double epochAccuracy = 0;
    int i, j, imageNum, correctClassifications;
    double modelConfidence;
    int modelPrediction;
    bool correctClassificationFlag;

    correctClassifications = 0;
    for (i = 0; i < testExNum; i++)
    {
        imageNum = rand() % 10000;
        forward_Prop(imageNum, true);
        modelConfidence = 0.0;
        modelPrediction = 0;
        
        correctClassificationFlag = false;
        for (j = 0; j < 10; j++)
        {
            if (output_Layer[j] > modelConfidence)
            {
                modelPrediction = j;
                modelConfidence = output_Layer[j];
            }
        }
        if (modelPrediction == test_label[imageNum]) //possibly remove if by subtracting prediction from label
        {
            correctClassificationFlag = true;
        }
        correctClassifications = correctClassifications + correctClassificationFlag; //+1 if correct
    }
    epochAccuracy = (correctClassifications / (double)testExNum) * 100;
    return epochAccuracy;
}
