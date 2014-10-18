#include "stdafx.h"
#include "video.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// This is mostly to test OpenCV integration and interface

class OpenCVImageReader : public IVideoReader
{
	int width;
	int height;

	cv::Mat continuousRGBA;
public:

	OpenCVImageReader(const std::string& filename);
	~OpenCVImageReader();

	virtual bool is_loaded()      { return /*loaded if*/continuousRGBA.data != 0; }

	virtual int  get_width()      { return width;  }
	virtual int  get_height()     { return height; }				   
											     
	virtual int  get_num_frames() { return 1; }
	virtual void seek(int frame)  { /* ignore */ }

	virtual void* get_next_frame();
};

OpenCVImageReader::OpenCVImageReader(const std::string& filename)
{
	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);

	// http://stackoverflow.com/questions/10265125/opencv-2-3-convert-mat-to-rgba-pixel-array
	continuousRGBA = cv::Mat(image.size(), CV_8UC4);
	cv::cvtColor(image, continuousRGBA, CV_BGR2RGBA);

	width  = continuousRGBA.size().width;
	height = continuousRGBA.size().height;
}

OpenCVImageReader::~OpenCVImageReader()
{
}

void* OpenCVImageReader::get_next_frame()
{
	return (void*)continuousRGBA.data;
}

IVideoReader* make_opencv_image_reader(const std::string& filename)
{
	return new OpenCVImageReader(filename);
}
