#ifndef __VIDEO_H__
#define __VIDEO_H__

#include <string>

// ====================================================================
// Interface for video reader
class IVideoReader
{
public:
	virtual bool  is_loaded()      = 0;
				  
	virtual int   get_width()      = 0;
	virtual int   get_height()	   = 0;
				 					
	virtual int   get_num_frames() = 0;
	virtual void  seek(int frame)  = 0;

	virtual void* get_next_frame() = 0;
	virtual void* get_this_frame() = 0;	// Returns the same data as the last call to get_next_frame
};

// ====================================================================
// Factory methods to make video readers
IVideoReader* make_opencv_image_reader(const std::string& filename);
IVideoReader* make_opencv_video_reader(const std::string& filename);

#endif