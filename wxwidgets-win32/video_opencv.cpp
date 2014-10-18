#include "stdafx.h"
#include "video.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class OpenCVVideoReader : public IVideoReader
{
	int width;
	int height;
	int frame_number;

	cv::VideoCapture cap;

	cv::Mat frame;
	cv::Mat continuousRGBA;

public:

	OpenCVVideoReader(const std::string& filename);
	~OpenCVVideoReader();

	virtual bool is_loaded()      { return cap.isOpened(); }

	virtual int  get_width()      { return width;  }
	virtual int  get_height()     { return height; }				   
											     
	virtual int  get_num_frames() { return (int)cap.get(CV_CAP_PROP_FRAME_COUNT); }
	virtual void seek(int frame);

	virtual void* get_next_frame();
	virtual void* get_this_frame() { return (void*)continuousRGBA.data; };
};

OpenCVVideoReader::OpenCVVideoReader(const std::string& filename)
{
	// http://stackoverflow.com/questions/19292052/read-a-sequence-of-frames-using-opencv-c
	
	bool video_opened = cap.open(filename);

	width  = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	continuousRGBA = cv::Mat(cv::Size(width, height), CV_8UC4);

	// Get first frame so that get_this_frame has a valid state
	frame_number = get_num_frames();
	get_next_frame();
}

OpenCVVideoReader::~OpenCVVideoReader()
{
}

void* OpenCVVideoReader::get_next_frame()
{
	frame_number++;
	if (frame_number >= get_num_frames()) {
		seek(0);
	}

    cap >> frame;
	cv::cvtColor(frame, continuousRGBA, CV_BGR2RGBA);

	return (void*)continuousRGBA.data;
}

void OpenCVVideoReader::seek(int frame)
{
	assert(frame < get_num_frames());
	cap.set(CV_CAP_PROP_POS_FRAMES, (double)frame);
	frame_number = frame;
}

IVideoReader* make_opencv_video_reader(const std::string& filename)
{
	return new OpenCVVideoReader(filename);
}


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
	virtual void* get_this_frame() { return (void*)continuousRGBA.data; };
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
