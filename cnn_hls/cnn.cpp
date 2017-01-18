//top function for mnist cnn
#include <math.h>
#include <iostream>
#include "cnn.h"

using namespace std;
//functions need to be defined:
//void mnist_cnn()
//float max_2()
//float max_4()
//float softmax()

/*----------------------------------------------
layer_names:
layer 0: convolution2d_1
layer 1: activation_1
layer 2: convolution2d_2
layer 3: activation_2
layer 4: maxpooling2d_1
layer 5: dropout_1
layer 6: flatten_1
layer 7: dense_1
layer 8: activation_3
layer 9: dropout_2
layer 10: dense_2
layer 11: activation_4
----------------------------------------------*/
void mnist_cnn(float weights_0[288], float biases_0[32],
			   float weights_2[9216], float biases_2[32],
			   float weights_7[589824], float biases_7[128],
			   float weights_10[1280], float biases_10[10],
			   float im_in[784],float final_out[10]){

	//parameters
	int stride = 1;
	int pool_stride = 2;

	//initialize 1st layer's output(convolution 1)
	float layer_out_0_fm[26][26][32];
	for(int i=0;i<26;i++){
		for(int j=0;j<26;j++){
			for(int k=0;k<32;k++){
				layer_out_0_fm[i][j][k] = biases_0[k];
			}
		}
	}
	//compute 1st layer's output(convolution 1)
	for(int out_dep=0;out_dep<32;out_dep++){
		for(int row=0;row<26;row++){
			for(int col=0;col<26;col++){
				for(int in_dep=0;in_dep<1;in_dep++){
					for(int i=0;i<3;i++){
						for(int j=0;j<3;j++){
							 layer_out_0_fm[row][col][out_dep] +=
							 weights_0[((i*3+j)*1+in_dep)*32+out_dep]*im_in[((stride*row+i)*28+(stride*col+j))*1+in_dep];
						}
					}
				}
			}
		}
	} //end of 1st layer:conv 1

	//debug only
/*	for(int out_dep=0;out_dep<32;out_dep++){
		for(int row=0;row<26;row++){
			for(int col=0;col<26;col++){
				cout<<layer_out_0_fm[row][col][out_dep]<<"\n";
			}
		}
	}
*/
	//2nd layer, RELU activation
	float layer_out_1_fm[26][26][32];
	for(int out_dep=0;out_dep<32;out_dep++){
		for(int row=0;row<26;row++){
			for(int col=0;col<26;col++){
				if(layer_out_0_fm[row][col][out_dep] < 0)
					layer_out_1_fm[row][col][out_dep] = 0;
				else
					layer_out_1_fm[row][col][out_dep] = layer_out_0_fm[row][col][out_dep];
			}
		}
	} //end of 2nd layer:activation 1
	
	//initialize 3rd layer's output(convolution 2)
	float layer_out_2_fm[24][24][32];
	for(int i=0;i<24;i++){
		for(int j=0;j<24;j++){
			for(int k=0;k<32;k++){
				layer_out_2_fm[i][j][k] = biases_2[k];
			}
		}
	}
	//compute 3rd layer's output(convolution 2)
	for(int out_dep=0;out_dep<32;out_dep++){
		for(int row=0;row<24;row++){
			for(int col=0;col<24;col++){
				for(int in_dep=0;in_dep<32;in_dep++){
					for(int i=0;i<3;i++){
						for(int j=0;j<3;j++){
							 layer_out_2_fm[row][col][out_dep] += 
							 weights_2[((i*3+j)*32+in_dep)*32+out_dep]*layer_out_1_fm[stride*row+i][stride*col+j][in_dep];
						}
					}
				}
			}
		}
	} //end of 3rd layer:conv 2

	//4th layer, RELU activation
	float layer_out_3_fm[24][24][32];
	for(int out_dep=0;out_dep<32;out_dep++){
		for(int row=0;row<24;row++){
			for(int col=0;col<24;col++){
				if(layer_out_2_fm[row][col][out_dep] < 0)
					layer_out_3_fm[row][col][out_dep] = 0;
				else
					layer_out_3_fm[row][col][out_dep] = layer_out_2_fm[row][col][out_dep];
			}
		}
	} //end of 4th layer:activation 2
	
	//5th layer, max pooling
	float layer_out_4_fm[12][12][32];
	for(int row=0;row<12;row++){
		for(int col=0;col<12;col++){
			for(int out_dep=0;out_dep<32;out_dep++){
				layer_out_4_fm[row][col][out_dep] = max_4(layer_out_3_fm[pool_stride*row+0][pool_stride*col+0][out_dep],
																				   layer_out_3_fm[pool_stride*row+0][pool_stride*col+1][out_dep],
                                                                                   layer_out_3_fm[pool_stride*row+1][pool_stride*col+0][out_dep],
			                                                                       layer_out_3_fm[pool_stride*row+1][pool_stride*col+1][out_dep]);
			}
		}
	} //end of 5th layer:max pooling
	
	//initialize 8th layer output(dense 1)
	float layer_out_7_fm[128];
	for(int i=0;i<128;i++){
		layer_out_7_fm[i] = biases_7[i];
		//calculate 8th layer output, dense 1 
		for(int row=0;row<12;row++){
			for(int col=0;col<12;col++){
				for(int out_dep=0;out_dep<32;out_dep++){
					layer_out_7_fm[i] += layer_out_4_fm[row][col][out_dep]*weights_7[((row*12+col)*32+out_dep)*128+i];
				}
			}
		} 
	}//end of 8th layer:dense 1
	
	//initialize 9th layer output
	float layer_out_8_fm[128];
	for(int i=0;i<128;i++){
		if(layer_out_7_fm[i]<0)
			layer_out_8_fm[i]=0;
		else
			layer_out_8_fm[i] = layer_out_7_fm[i];
	}//end of 9th layer, RELU activation
	
	//11th layer, dense
	float layer_out_10_fm[10];
	for(int i=0;i<10;i++){
		layer_out_10_fm[i] = biases_10[i];
		for(int j=0;j<128;j++){
			layer_out_10_fm[i] += layer_out_8_fm[j]*weights_10[j*10+i];		
		}
	}//end of 11th layer
	
	//12th layer, activation 4, softmax
	float layer_out_11_fm[10];
	softmax(layer_out_10_fm,layer_out_11_fm);
	//end of 12th layer

	for(int i=0;i<10;i++){
		final_out[i]=layer_out_11_fm[i];
	}

}//end of mnist_cnn

float max_2(float a, float b){
	if(a>b)
		return a;
	else 
		return b;
}

float max_4(float a, float b, float c, float d){
	float temp_1,temp_2;
	temp_1 = max_2(a,b);
	temp_2 = max_2(c,d);
	return max_2(temp_1,temp_2);
}

void softmax(float num[10], float num_out[10]){
	float sum = 0;
	float e_num[10];
	for(int i=0;i<10;i++){
		e_num[i] = expf(num[i]);
		sum += e_num[i];
	}
	
	for(int i=0;i<10;i++){
		num_out[i] = e_num[i]/sum; 		
	}
}
