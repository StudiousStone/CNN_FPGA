#include <iostream>
#include <fstream>
#include "cnn.h"

using namespace std;

int main(){

	cout<<"begin to test"<<"\n";
	static float im_in[784];
	static float weights_0[288];
	static float biases_0[32];
	static float weights_2[9216];
	static float biases_2[32];
	static float weights_7[589824];
	static float biases_7[128];
	static float weights_10[1280];
	static float biases_10[10];
    
	//begin to read file into testbench
	cout<<"reading input image file"<<"\n";
	std::ifstream image_in("D:\\HLS_Prj\\mnist_cnn\\input_3.txt");
	if (image_in.good()) {
		for(int i=0; i<784; i++){
			image_in >> im_in[i];
			//cout<<im_in[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		image_in.close();
	} else {
		printf("Read input_3.txt failed!\n");
	}
	
	cout<<"reading weights 0 file"<<"\n";
	std::ifstream weighs_0_in("D:\\HLS_Prj\\mnist_cnn\\weights_0.txt");
	if (weighs_0_in.good()) {
		for(int i=0; i<784; i++){
			weighs_0_in >> weights_0[i];
			//cout<<weights_0[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		weighs_0_in.close();
	} else {
		printf("Read weights_0.txt failed!\n");
	}
	
	cout<<"reading weights 2 file"<<"\n";
	std::ifstream weighs_2_in("D:\\HLS_Prj\\mnist_cnn\\weights_2.txt");
	if (weighs_2_in.good()) {
		for(int i=0; i<9216; i++){
			weighs_2_in >> weights_2[i];
			//cout<<weights_2[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		weighs_2_in.close();
	} else {
		printf("Read weights_2.txt failed!\n");
	}

	cout<<"reading weights 7 file"<<"\n";
	std::ifstream weighs_7_in("D:\\HLS_Prj\\mnist_cnn\\weights_7.txt");
	if (weighs_7_in.good()) {
		for(int i=0; i<589824; i++){
			weighs_7_in >> weights_7[i];
			//cout<<weights_7[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		weighs_7_in.close();
	} else {
		printf("Read weights_7.txt failed!\n");
	}

	cout<<"reading weights 10 file"<<"\n";
	std::ifstream weighs_10_in("D:\\HLS_Prj\\mnist_cnn\\weights_10.txt");
	if (weighs_10_in.good()) {
		for(int i=0; i<1280; i++){
			weighs_10_in >> weights_10[i];
			//cout<<weights_10[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		weighs_10_in.close();
	} else {
		printf("Read weights_10.txt failed!\n");
	}	

	cout<<"reading biases_0.txt"<<"\n";
	std::ifstream biases_0_in("D:\\HLS_Prj\\mnist_cnn\\biases_0.txt");
	if (biases_0_in.good()) {
		for(int i=0; i<1280; i++){
			biases_0_in >> biases_0[i];
			//cout<<weights_10[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		biases_0_in.close();
	} else {
		printf("Read biases_0.txt failed!\n");
	}
	
	cout<<"reading biases_2.txt"<<"\n";
	std::ifstream biases_2_in("D:\\HLS_Prj\\mnist_cnn\\biases_2.txt");
	if (biases_2_in.good()) {
		for(int i=0; i<1280; i++){
			biases_2_in >> biases_2[i];
			//cout<<weights_10[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		biases_2_in.close();
	} else {
		printf("Read biases_2.txt failed!\n");
	}
	
	cout<<"reading biases_7.txt"<<"\n";
	std::ifstream biases_7_in("D:\\HLS_Prj\\mnist_cnn\\biases_7.txt");
	if (biases_7_in.good()) {
		for(int i=0; i<1280; i++){
			biases_7_in >> biases_7[i];
			//cout<<weights_10[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		biases_7_in.close();
	} else {
		printf("Read biases_7.txt failed!\n");
	}
	
	cout<<"reading biases_10.txt"<<"\n";
	std::ifstream biases_10_in("D:\\HLS_Prj\\mnist_cnn\\biases_10.txt");
	if (biases_10_in.good()) {
		for(int i=0; i<1280; i++){
			biases_10_in >> biases_10[i];
			//cout<<weights_10[i]<<"\n";
		}
		cout << "read suceed."<<"\n";
		biases_10_in.close();
	} else {
		printf("Read biases_10_in.txt failed!\n");
	}
	//end of reading files
	
	//begin to test dut
	float final_out[10];
	mnist_cnn(weights_0,biases_0,
			  weights_2, biases_2,
		      weights_7, biases_7,
			  weights_10, biases_10,
			  im_in, final_out);
	

	cout<<"final output: "<<"\n";
	for(int i=0;i<10;i++){
		cout<<i<<"th output:"<<final_out[i]<<"\n";
	}

	return 0;
}
