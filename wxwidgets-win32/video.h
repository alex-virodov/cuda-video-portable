#ifndef __VIDEO_H__
#define __VIDEO_H__

#include <string>

class IVideoReader
{
public:
	virtual bool  is_loaded()      = 0;
				  
	virtual int   get_width()      = 0;
	virtual int   get_height()	   = 0;
				 					
	virtual int   get_num_frames() = 0;
	virtual void  seek(int frame)  = 0;

	virtual void* get_next_frame() = 0;
};

IVideoReader* make_opencv_image_reader(const std::string& filename);
IVideoReader* make_opencv_video_reader(const std::string& filename);

#endif