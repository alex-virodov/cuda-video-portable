#include <assert.h>
#include <iostream>
#include <string>
#include <cmath>
#include <stdio.h>

#ifdef _MSC_VER
#include <Windows.h>	// needed for gl interop
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "vector_types.h"
#include "me.h"



// ====================================================================
// Assert functionality
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuAssert(ans) { gpuAssertImpl((ans), __FILE__, __LINE__); }
inline void gpuAssertImpl(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (true) exit(code);
   }
}

// ====================================================================
// CUDA class to track state/resources
class CUDAMotionEstimation : public IMotionEstimation
{
	int n_frames;

	uchar4** frames_device;	
	uchar4*  result_device;
	uchar4*  result_host;   // TODO: PBO, OpenGL interop

	dim3     dim_threads_per_block;
	dim3     dim_blocks;

public:
	CUDAMotionEstimation(int width, int height);
	~CUDAMotionEstimation();

	virtual void load_frame(int frame, void* rgba_data);

	virtual void estimate();
												           
	virtual void store_result(void* rgba_data);
	virtual void store_result(GLuint tex_id);
};

// ====================================================================
CUDAMotionEstimation::CUDAMotionEstimation(int width, int height)
: IMotionEstimation(width, height)
{
	n_frames = 2;
	frames_device = new uchar4*[n_frames];

	for (int i = 0; i < n_frames; i++) {
		cudaMalloc( &frames_device[i], width*height*4);
	}

	cudaMalloc( &result_device, width*height*4);

	result_host = new uchar4[width*height];

	// Use 64 threads constant. May tweak later
	dim_threads_per_block = dim3(8, 8);

	// Ensure image coverage. Kernel will have checks for
	// out-of-bounds.
	dim_blocks = dim3((int)ceil((double)width / 8.0), (int)ceil((double)height / 8.0));
}

// ====================================================================
CUDAMotionEstimation::~CUDAMotionEstimation()
{
	cudaFree(result_device);

	for (int i = 0; i < n_frames; i++) { cudaFree(frames_device[i]); }
	delete frames_device;
	delete result_host;
}


// ====================================================================
void CUDAMotionEstimation::load_frame(int frame, void* rgba_data)
{
	assert(frame < n_frames);

	gpuAssert( cudaMemcpy(
		/*dst =*/frames_device[frame], 
		/*src =*/rgba_data, 
		/*size=*/width*height*4,
		cudaMemcpyHostToDevice));
}

// ====================================================================
void CUDAMotionEstimation::store_result(void* rgba_data)
{
	gpuAssert( cudaMemcpy(
		/*dst =*/rgba_data, 
		/*src =*/result_device, 
		/*size=*/width*height*4,
		cudaMemcpyDeviceToHost));
}

// ====================================================================
void CUDAMotionEstimation::store_result(GLuint tex_id)
{
	// TODO: PBO, OpenGL interop. now copying back to host. inefficient.
	store_result(result_host);

	glBindTexture  (GL_TEXTURE_2D, tex_id );
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, result_host);
	glBindTexture  (GL_TEXTURE_2D, tex_id );
}

// ====================================================================
// CUDA kernel to do motion estimation. For now just does image 
// differencing
__global__ void cuda_me_kernel(
	const int width,
	const int height,
	uchar4* result, uchar4* frame1, uchar4* frame2
	)
{
	const unsigned int x = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    const unsigned int y = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	if (x < width && y < height) {
		const int idx = x + y * width;
		result[idx].x = abs(frame1[idx].x - frame2[idx].x);
		result[idx].y = abs(frame1[idx].y - frame2[idx].y);
		result[idx].z = abs(frame1[idx].z - frame2[idx].z);
	}
}

// ====================================================================
void CUDAMotionEstimation::estimate()
{
	cuda_me_kernel<<< this->dim_blocks, this->dim_threads_per_block>>>( 
		width, height, result_device, frames_device[0], frames_device[1]);

	gpuAssert( cudaPeekAtLastError() );
}


// ====================================================================
// factory method to construct cuda instance
IMotionEstimation* make_me_cuda(int width, int height)
{
	return new CUDAMotionEstimation(width, height);
}













