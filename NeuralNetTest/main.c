#include "include/mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//3 hidden layers with 10 neurons, 784 inputs, 10 outputs (1 -> 2 -> 3 -> output)

double output_Layer[10] = { 0 };
double Layer_1[10] = { 0 };
double Layer_2[10] = { 0 };
double Layer_3[10] = { 0 };

double weights_Layer_1[784][10] = { 0 };
double weights_Layer_2[10][10] = { 0 };
double weights_Layer_3[10][10] = { 0 };
double weights_Output[10][10] = { 0 };

double bias_Layer_1[10] = { 0 };
double bias_Layer_2[10] = { 0 };
double bias_Layer_3[10] = { 0 };
double bias_Output[10] = { 0 };


int main(void)
{
    // call to store mnist in array
    load_mnist();
    //initialise weight matrices (-0.1 to 0.1) todo: make range parameters
    initialise_Weight_Matrices();
    //initialise bias matrices (-0.1 to 0.1)
    initialise_Bias_Matrices();
    // print pixels of first data in test dataset
    int i;
    for (i = 0; i < 784; i++) {
        printf("%1.1f ", test_image[0][i]);
        if ((i + 1) % 28 == 0) putchar('\n');
    }

    // print first label in test dataset
    printf("label: %d\n", test_label[0]);

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
    srand(time(NULL));
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
    srand(time(NULL));
    for (x = 0; x < 10; x++)
    {
        bias_Layer_1[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        bias_Layer_2[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        bias_Layer_3[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
        bias_Output[x] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
    }
}