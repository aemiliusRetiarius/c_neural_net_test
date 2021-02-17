#include "include/mnist.h"

//3 hidden layers with 10 neurons, 784 inputs, 10 outputs

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