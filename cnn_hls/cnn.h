#ifndef CNN_H__
#define CNN_H__

#include <iostream>
#include <math.h>

void mnist_cnn(float weights_0[288], float biases_0[32],
			   float weights_2[9216], float biases_2[32],
			   float weights_7[589824], float biases_7[128],
			   float weights_10[1280], float biases_10[10],
			   float im_in[784],float final_out[10]);

float max_2(float a, float b);

float max_4(float a, float b, float c, float d);

void softmax(float num[10], float num_out[10]);

#endif
