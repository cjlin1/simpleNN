function M = gpu(fun_arr, dim)

global gpu_use
global float_type

if gpu_use
	if isa(fun_arr,'function_handle')
		M = fun_arr(dim, float_type, 'gpuArray');
	else
		M = gpuArray(fun_arr);
	end
else
	if isa(fun_arr,'function_handle')
		M = fun_arr(dim, float_type);
	else
		M = fun_arr;
	end
end
