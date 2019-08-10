function M = array(M)

global gpu_use
if gpu_use == 1
	M = gpuArray(M);
end
