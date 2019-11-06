function M = ftype(M)

global float_type
if float_type == 'single'
	M = single(M);
end
