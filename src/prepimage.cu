#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "prepimage.h"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)



__global__ void image_kernel(unsigned char* input, float3* output, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const int px = input[y * width + 3 * x];

	const unsigned char blue	= input[px];
	const unsigned char green	= input[px + 1];
	const unsigned char red		= input[px + 2];

	float3 fpx = make_float3(blue, green, red);
	/*
	fpx.x = (fpx.x - 127f) / 127f;
	fpx.y = (fpx.y - 127f) / 127f;
	fpx.z = (fpx.z - 127f) / 127f;*/

	output[y*width+x] = fpx;
}



cudaError_t prepimage(const cv::Mat& input, float3* output)
{
	/*
	if (input == NULL || output == NULL)
		return cudaErrorInvalidDevicePointer;*/

	size_t width = input.step;
	size_t height = input.rows;

	if (width == 0 || height == 0)
		return cudaErrorInvalidValue;

	unsigned char *d_input;
	const int inBytes = input.step * input.rows; // step includes the 3 channel count

	cudaMalloc<unsigned char>(&d_input, inBytes);
	cudaMemcpy(d_input, input.ptr(), inBytes, cudaMemcpyHostToDevice);



	// launch kernel
	const dim3 block(8, 8);
	//const dim3 grid(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);


	image_kernel<<<grid, block>>>(d_input, output, width, height);

	return cudaGetLastError();
}
