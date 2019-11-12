#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

template<class Tval>
void __global__ mex_accum(const double *idx, const Tval *val, unsigned long long int const N, Tval *vTP)
{
	unsigned long long int i = blockDim.x * blockIdx.x + threadIdx.x;

	while (i < N)
	{
		unsigned long long int vTP_idx = idx[i] - 1;
		if(val[i] != 0.0)
			atomicAdd(&vTP[vTP_idx], val[i]);
		i += blockDim.x*gridDim.x;
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{

	const mxGPUArray *idx, *val;
	mxGPUArray *vTP;

	mxInitGPU();

	idx = mxGPUCreateFromMxArray(prhs[0]);
	val = mxGPUCreateFromMxArray(prhs[1]);

	unsigned long long int const Nout = mxGetScalar(prhs[2]);
	mwSize dims[1] = {Nout};
	vTP = mxGPUCreateGPUArray(1, dims, mxGPUGetClassID(val), mxREAL, MX_GPU_INITIALIZE_VALUES);

	unsigned long long int const N = mxGPUGetNumberOfElements(val);
	const int threadsPerBlock = 256;
	unsigned long long int blocksPerGrid;
	blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	const double *d_idx;
	d_idx = (const double*) (mxGPUGetDataReadOnly(idx));
	switch (mxGPUGetClassID(val))
	{
		case mxDOUBLE_CLASS:
			const double *d_val;
			double *d_vTP;
			d_val = (const double*) (mxGPUGetDataReadOnly(val));
			d_vTP = (double*) (mxGPUGetData(vTP));
			mex_accum<<<blocksPerGrid, threadsPerBlock>>>(d_idx, d_val, N, d_vTP);
			break;
		case mxSINGLE_CLASS:
			const float *f_val;
			float *f_vTP;
			f_val = (const float*) (mxGPUGetDataReadOnly(val));
			f_vTP = (float*) (mxGPUGetData(vTP));
			mex_accum<<<blocksPerGrid, threadsPerBlock>>>(d_idx, f_val, N, f_vTP);
			break;
	}
	plhs[0] = mxGPUCreateMxArrayOnGPU(vTP);

	mxGPUDestroyGPUArray(idx);
	mxGPUDestroyGPUArray(val);
	mxGPUDestroyGPUArray(vTP);
}

