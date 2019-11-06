function make(cudabin_path)
% /usr/local/cuda-8.0_back/bin or /usr/local/cuda/bin

try
	cd cnn;
	setenv('MW_NVCC_PATH',cudabin_path);
	mexcuda('accum.cu');
	cd ..;
catch err
	disp(err.message);
	cd ..;
end
