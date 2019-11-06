function M = gpu(M)

global gpu_use
if gpu_use
	M = gpuArray(M);
end
