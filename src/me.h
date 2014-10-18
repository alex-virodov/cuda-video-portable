#ifndef __ME_H__
#define __ME_H__

#include <GL/gl.h>
#include <limits>

// ====================================================================
// Interface for motion estimation functionality
class IMotionEstimation
{
protected:
	int width;
	int height;
public:

	IMotionEstimation(int width, int height)
		: width(width), height(height) {}

	virtual void load_frame(int frame, void* rgba_data)    = 0;

	virtual void estimate()                                = 0;
												           
	virtual void store_result(void* rgba_data)             = 0;
	virtual void store_result(GLuint tex_id)               = 0;
};

void me_test_cpu_sad();
void me_test_cpu_list();

IMotionEstimation* make_me_cpu (int width, int height);
IMotionEstimation* make_me_cuda(int width, int height);

typedef unsigned int me_dist;
const me_dist max_dist = std::numeric_limits<me_dist>::max();

// ====================================================================
// Use a constant size buffer for candidates. If a candidate
// is smaller than any one entry in the buffer, push the largest
// entry out.
class MECandidateList
{
	// All fields directly accessible for speed/simplicity

	// TODO: Use 'signed char' for xs (store delta x) to conserve GPU memory when ported
	// TODO: not sure 'min' tracking helps. Will query min only once at the end.
public:
	static const int list_size = 4;
	
	me_dist dists [list_size];
	int     xs    [list_size];
	int     ys    [list_size];
	me_dist min;				// keep track of smallest distance to optimize min searches
	me_dist max;				// keep track of largest distance to optimize inserts

	MECandidateList() 
	{ 
		for (int i = 0; i < list_size; i++) { dists[i] = max_dist; }
		min = max_dist;
		max = max_dist;
	}

	void insert(me_dist dist, int x, int y)
	{
		// if this one is larger than the largest element,
		// no need to insert
		if (dist >= max) { return; }
		
		// insert this element instead of the largest
		// assuming small list - do lienar searches. Compiler should unroll loops.
		// TODO: in cuda, does 'break' make sense or even legal in this kind of loop?
		int max_idx = -1;
		for (int i = 0; i < list_size; i++) { if (dists[i] == max) { max_idx = i; break; } }

		assert(max_idx >= 0 && "MECandidateList::max is inconsistent");

		dists[max_idx] = dist;
		xs   [max_idx] = x;
		ys   [max_idx] = y;

		// update tracking. note that min property is changing only when we insert new
		// smallest element, so no need for loops. That's not the case for max property, because
		// we remove the largest element, and we don't know what's the next largest is.
		if (dist < min) { min = dist; }

		max = dist;
		for (int i = 0; i < list_size; i++) { if (dists[i] > max) { max = dists[i]; } }
	}

	int get_min_idx()
	{
		for (int i = 0; i < list_size; i++) { if (dists[i] == min) { return i; } }
		
		assert("MECandidateList::min is inconsistent" && 0);
		return 0;
	}
};

#endif
