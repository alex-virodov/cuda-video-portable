#include "stdafx.h"
#include "me.h"
#include <memory>
#include <vector>
#include <limits>
#include <iostream>

using namespace std;

// Motion estimation implementation on CPU
class CPUMotionEstimation : public IMotionEstimation
{
	int n_frames;

	unsigned char** frames;	
	unsigned char*  result;

public:
	CPUMotionEstimation(int width, int height)
	: IMotionEstimation(width, height)
	{
		n_frames = 2;
		frames = new unsigned char*[n_frames];
		
		for (int i = 0; i < n_frames; i++) {
			frames[i] = new unsigned char[width*height*4];
		}

		result = new unsigned char[width*height*4];
	}

	~CPUMotionEstimation()
	{
		for (int i = 0; i < n_frames; i++) { delete frames[i]; }
		delete frames;
		delete result;
	}

	virtual void load_frame(int frame, void* rgba_data);

	virtual void estimate();
												           
	virtual void store_result(void* rgba_data);
	virtual void store_result(GLuint tex_id);
};

void CPUMotionEstimation::load_frame(int frame, void* rgba_data)
{
	assert(frame < n_frames);

	memcpy(frames[frame], rgba_data, width*height*4);
}

// Implement as "kernels" - per-pixel operations, to ease
// porting into CUDA/OpenCL

// ====================================================================
// Compute Sum Absolute Distance between a given marker 
// (centered at x_marker,y_marker) in frame1 and another location (centered 
// at x_scan,y_scan) in frame2
// TODO: Handle edges by not summing up differences on missing pixels?
me_dist  cpu_me_kernel_rgba_sad(
	const int width, 
	unsigned char* frame1, 
	unsigned char* frame2, 
	const int x_marker, 
	const int y_marker,
	const int x_scan, 
	const int y_scan,
	const int marker_size)
{
	const int half_marker = (marker_size-1)/2;

	me_dist dist = 0;

	for (int j = -half_marker; j <= +half_marker; j++) 
	{
		for (int i = -half_marker; i <= +half_marker; i++) 
		{
			const int marker_idx = ((x_marker+i) + (y_marker+j)*width)*4;
			const int scan_idx   = ((x_scan  +i) + (y_scan  +j)*width)*4;

			// Compute the distance in 3 color components
			for (int k = 0; k < 3; k++) {
				dist += abs(frame1[marker_idx+k] - frame2[scan_idx+k]);
			}
		}
	}

	return dist;
}

void cpu_me_kernel_rgba(
	const int width, 
	const int height, 
	unsigned char* result, 
	unsigned char* frame1, 
	unsigned char* frame2, 
	const int i, 
	const int j,
	const int marker_size, 
	const int window_size)
{
	// ASSUMING me_window_size, me_marker_size is odd

	assert(i < width);
	assert(j < height);

	const int half_marker = (marker_size-1)/2;
	const int result_idx  = (i +j*width)*4;

	// For edge pixels there is no full marker, so set
	// them to some predefined value
	if (i <        half_marker || j <         half_marker || 
		i >= width-half_marker || j >= height-half_marker)
	{
		// TODO: for now set to 0xAA for debugging. Change to 0xFF? Or zero?
		result[result_idx+0] = result[result_idx+1] = 
			result[result_idx+2] = result[result_idx+3] = 0xAA;
		// TODO: cuda - is return better than computing but not setting?
		// also see: http://stackoverflow.com/questions/14869513/divergence-in-cuda-exit-from-a-thread-in-kernel
		return;
	}

	// Priority list of candidates by similarity distance (not euclidian distance)
	MECandidateList candidates;

	// Compute scanning bounds and adjust to window size
	int xscan_min = i - (window_size-1)/2 + half_marker;
	int xscan_max = i + (window_size-1)/2 - half_marker;
	int yscan_min = j - (window_size-1)/2 + half_marker;
	int yscan_max = j + (window_size-1)/2 - half_marker;

	if (xscan_min <         half_marker) { xscan_min = half_marker; }
	if (xscan_max >=  width-half_marker) { xscan_max = width-half_marker-1; }
	if (yscan_min <         half_marker) { yscan_min = half_marker; }
	if (yscan_max >= height-half_marker) { yscan_max = height-half_marker-1; }

	// Scan the window for the marker
	for (int y = xscan_min; y <= xscan_max; y++) 
	{
		for (int x = xscan_min; x <= xscan_max; x++) 
		{
			me_dist dist = cpu_me_kernel_rgba_sad(
					width, frame1, frame2, 
					/*x_marker=*/i, /*y_marker=*/j,
					/*x_scan  =*/x, /*y_scan  =*/y, marker_size);

			candidates.insert(dist, x, y);
		}
	}

	// Get the smallest (in distance) candidate and update the
	// result image with information about its location
	int min_idx = candidates.get_min_idx();

	result[result_idx+0] = candidates.dists[min_idx];
	result[result_idx+1] = 0*candidates.xs   [min_idx];
	result[result_idx+2] = 0*candidates.ys   [min_idx];
	result[result_idx+3] = 0;
}

void CPUMotionEstimation::estimate()
{
	// ASSUMING frames were loaded

	for (int j = 0; j < height; j++) 
	{
		for (int i = 0; i < width; i++) 
		{
#if 0
			cpu_me_kernel_rgba(width, height, result, frames[0], frames[1], i, j,
				/*me_marker_size=*/3, /*me_window_size=*/11);
#else
			const int idx = (i + j*width)*4;

			for (int k = 0; k < 4; k++) {
				result[idx+k] = (unsigned char)abs((int)frames[0][idx+k] - (int)frames[1][idx+k]);
			}
#endif
		}
	}
}
									           
void CPUMotionEstimation::store_result(void* rgba_data)
{
	memcpy(rgba_data, result, width*height*4);
}

void CPUMotionEstimation::store_result(GLuint tex_id)
{
	// Upload texture (TODO: use PBO?)

	glBindTexture  (GL_TEXTURE_2D, tex_id );
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, result);
	glBindTexture  (GL_TEXTURE_2D, tex_id );
}

IMotionEstimation* make_me_cpu (int width, int height)
{
	return new CPUMotionEstimation(width, height);
}

void me_test_cpu_list()
{
	MECandidateList list1;
	MECandidateList list2;
	MECandidateList list3;

	list1.insert( 99,   99, 10);
	list1.insert( 98,   98, 10);
	list1.insert( 97,   97, 10);
	list1.insert( 96,   96, 10);
	list1.insert( 95,   95, 10);

	list2.insert( 95,   95, 10);
	list2.insert( 96,   96, 10);
	list2.insert( 97,   97, 10);
	list2.insert( 98,   98, 10);
	list2.insert( 99,   99, 10);

	list3.insert( 29,   29, 10);
	list3.insert( 17,   17, 10);
	list3.insert( 80,   80, 10);
	list3.insert( 57,   57, 10);
	list3.insert( 31,   31, 10);
	list3.insert(  7,    7, 10);
	list3.insert( 37,   37, 10);
	list3.insert( 40,   40, 10);
	list3.insert( 34,   34, 10);
	list3.insert( 88,   88, 10);

	// Checked in debugger. 
	// TODO: use CppUnit?

}

void me_test_cpu_sad()
{
	int data1[] = { 0x70, 0x80, 0x90,
				    0xA0, 0xB0, 0xC0,
				    0xD0, 0xE0, 0xF0,};

	// distance sum|data1-data2| = 1+2+3+4+5+6+7+8+9
	int data2[] = { 0x71, 0x82, 0x93,
				    0xA4, 0xB5, 0xC6,
				    0xD7, 0xE8, 0xF9,};

	int data3[] = { 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
				    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
				    0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
				    0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,};

	int data4[] = { 0x33, 0x34, 0x35, 0xAA, 0x36, 0x37, 0x38,
				    0x43, 0x44, 0x45, 0xAA, 0x46, 0x47, 0x48,
				    0x53, 0x54, 0x55, 0xAA, 0x56, 0x57, 0x58,
				    0x63, 0x64, 0x65, 0xAA, 0x66, 0x67, 0x68,};

	int data34_result[7*4];

	me_dist dist_data12 = 
		cpu_me_kernel_rgba_sad(/*width=*/3, (unsigned char*)data1, (unsigned char*)data2, 
			/*x_marker=*/1, /*y_marker=*/1,
			/*x_scan  =*/1, /*y_scan  =*/1, /*marker_size=*/3);

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 7; i++) {
			cpu_me_kernel_rgba(
				/*width =*/7, 
				/*height=*/4, 
				/*result=*/(unsigned char*)data34_result, 
				/*frame1=*/(unsigned char*)data3, 
				/*frame2=*/(unsigned char*)data4, 
				/*i=*/i, 
				/*j=*/j,
				/*marker_size=*/3, 
				/*window_size=*/5);
		}
	}

	int x = 0;

	// Checked in debugger. 
	// TODO: use CppUnit?
}