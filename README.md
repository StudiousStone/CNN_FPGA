# CNN_FPGA
Just for my personal research

---------------------------------------------------------------------------------------------------------------------------
~/python_rebuild_model/: rebuild the cnn model using python
Now we have small errors during each layer, the guess is that the error could be caused by:

	1. Precision problems, but in the code every number is 'float32'.
	2. The order each calculation is done perhaps matters.

The data of each picture can be read out through the code: CNN_FPGA/python_rebuild_model/mnist_dataset/
Normalization is needed after the extraction: x = x/255

The mnist_cnn_eg.py can generate and train a simple CNN then output the weights and biases in the file mnist_cnn_trained.h5.

The mnist_cnn_load.py can load the trained module and run the feedforward process by self-written code.

The mnist_cnn_hls.py is a version in which all the weights and inputs are flattened into 1-dimension arraies. It also saves the weights and biases into the folder 'mnist_weight_bias'.


--------------------------------------------------------------------------------------------------------------------------- ~/cnn_hls/: the Vivado High Level Synthesis code for the simple cnn
The simulation result is now the same as the python version.
Maybe we can compare the final results with the golden results to make the co-simulation work.
Next study what to do with the interfaces, how to optimize the module a bit so that it behaves like an accelerator.
