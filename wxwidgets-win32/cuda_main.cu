#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cmath>

__global__ void MatrixCopy( float* dst, float* src, unsigned int matrixRank )
{
	unsigned int x = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    unsigned int y = ( blockDim.y * blockIdx.y ) + threadIdx.y;
 
    unsigned int index = ( matrixRank * y ) + x;
    if ( index < matrixRank * matrixRank ) // prevent reading/writing array out-of-bounds.
    {
        dst[index] = src[index];
    }
}

int zz_main()
{
	const unsigned int matrixRank = 1025;
    unsigned int numElements = matrixRank * matrixRank;
    size_t size = ( numElements ) * sizeof(float);
 
    std::cout << "Matrix Size: " << matrixRank << " x " << matrixRank << std::endl;
    std::cout << "Total elements: " << numElements << std::endl;

	std::cout << "Allocating [host] buffers for source and destination matices..." << std::endl;
	// Allocate host memory to store matrices.
	float* matrixSrcHost = new float[numElements];
	float* matrixDstHost = new float[numElements];

	std::cout << "Initializing [host] buffers for source and destination matrices..." << std::endl;
	for ( unsigned int i = 0; i < numElements; ++i )
	{
		matrixSrcHost[i] = static_cast<float>(i); // Source matrix initialized to i;
		matrixDstHost[i] = 0.0f; // Destination matrix initialized to 0.0;
	}

	std::cout << "Allocating [device] buffers for source and destination matrices..." << std::endl;
	float* matrixSrcDevice;
	float* matrixDstDevice;
	cudaMalloc( &matrixSrcDevice, size );
	cudaMalloc( &matrixDstDevice, size );

	std::cout << "Initialize [device] buffers using [host] buffers..." << std::endl;
	cudaMemcpy( matrixSrcDevice, matrixSrcHost, size, cudaMemcpyHostToDevice );
	cudaMemcpy( matrixDstDevice, matrixDstHost, size, cudaMemcpyHostToDevice );

	// Maximum number of threads per block dimension (assuming a 2D thread block with max 512 threads per block).
	unsigned int maxThreadsPerBlockDim = min( matrixRank, 16U );
 
	std::cout << "Determine block and thread granularity for CUDA kernel..." << std::endl;
 
	size_t blocks = (size_t)ceilf( matrixRank / (float)maxThreadsPerBlockDim );
	dim3 blockDim( blocks, blocks, 1 );
	size_t threads = (size_t)ceilf( matrixRank / (float)blocks );
	dim3 threadDim( threads, threads, 1 );

	std::cout << "Invoke the kernel with block( " << blockDim.x << ", " << blockDim.y << ", 1 ), thread( " << threadDim.x << ", " << threadDim.y << ", 1 )." << std::endl;
	MatrixCopy<<< blockDim, threadDim >>>( matrixDstDevice, matrixSrcDevice, matrixRank );

	std::cout << "Copy resulting [device] buffers to [host] buffers..." << std::endl;
//	cudaMemcpy( matrixSrcHost, matrixSrcDevice, size, cudaMemcpyDeviceToHost );
	cudaMemcpy( matrixDstHost, matrixDstDevice, size, cudaMemcpyDeviceToHost );

	std::cout << "Verifying the result (source and destination matrices should now be the same)." << std::endl;
	bool copyVerified = true;
	for ( unsigned int i = 0; i < numElements; ++i )
	{
		if ( matrixDstHost[i] != matrixSrcHost[i] )
		{
			copyVerified = false;
			std::cerr << "Matrix destination differs from source:" << std::endl;
			std::cerr << "\tDst[" << i << "]: " << matrixDstHost[i] << " != " << "Dst[" << i << "]: " << matrixSrcHost[i] << std::endl;
		}
	}

	std::cout << "Free [device] buffers..." << std::endl;
	cudaFree( matrixSrcDevice );
	cudaFree( matrixDstDevice );
 
	std::cout << "Free [host] buffers..." << std::endl;
	delete [] matrixSrcHost;
	delete [] matrixDstHost;

	return 0;
}