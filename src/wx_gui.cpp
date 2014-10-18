#include "stdafx.h"
#include "GL/glu.h"
#include "glscene.h"

#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <cmath>

using namespace std;

// ====================================================================
// Application class
class MyApp: public wxApp
{

public:
	MyApp();

	virtual bool OnInit();
};

wxIMPLEMENT_APP_CONSOLE(MyApp);
//wxIMPLEMENT_APP(MyApp);

// ====================================================================
// Window class
class MyFrame: public wxFrame
{
public:
	boost::thread* draw_thread;

	wxGLCanvas*    canvas;
	wxBoxSizer*    canvas_sizer;

	MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

	void onClose(wxCloseEvent& event);

private:
	wxDECLARE_EVENT_TABLE();
};

wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
	EVT_CLOSE(MyFrame::onClose)
wxEND_EVENT_TABLE()

void thread_func(wxGLCanvas* canvas);

MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
: wxFrame(NULL, wxID_ANY, title, pos, size), draw_thread(NULL)
{
	// TODO: add sizer
	canvas = new wxGLCanvas(this, wxID_ANY, /*attribList=*/NULL, wxDefaultPosition, size);
		
	// TODO: Handle OpenGL thread updates on windowresizes.
	// TODO: Window resize on linux is blockingly slow.
	draw_thread = new boost::thread(thread_func, canvas);
}

void MyFrame::onClose(wxCloseEvent& event)
{
	// Terminate thread. Assuming the thread is
	// operating properly, so will wait for it indefinitely.
	if (draw_thread) {
		draw_thread->interrupt();
		draw_thread->join();
		delete draw_thread;
		draw_thread = 0;
	}

	Destroy();
}


// ====================================================================
// Drawing thread
void thread_func(wxGLCanvas* canvas)
{
	wxGLContext gl_context(canvas);

	gl_context.SetCurrent(*canvas);

	GLScene scene;

	try {
	while(true) 
	{
		boost::this_thread::interruption_point();

		// cout << ":: still running" << endl;

		// TODO: is canvas->GetSize() thread-safe? probably not...
		// TODO: call resize only on resize
		scene.resize(canvas->GetSize().x, canvas->GetSize().y);
		scene.render();

		//	cout << boost::chrono::duration_cast<boost::chrono::milliseconds>(frame_clock::now() - last_frame_time) << endl;
		//	last_frame_time = frame_clock::now();
			glFlush();
			canvas->SwapBuffers();

		// boost::this_thread::sleep(boost::posix_time::millisec(500));
	}
	} catch (...) {
		cout << "thread exit" << endl;
	}
}

// ====================================================================
// Application class implementation

MyApp::MyApp()
{
#ifndef _MSC_VER
	// On linux, need to initialize X threading properly.
	// https://forums.wxwidgets.org/viewtopic.php?t=32346&p=139431
	cout << ":: calling XInitThreads" << endl;
	int result = XInitThreads ();
	assert(result != 0);
#endif
}

bool MyApp::OnInit()
{
	MyFrame *frame = new MyFrame( "Video + CUDA", wxPoint(50, 50), wxSize(800, 600) );

	frame->Show( true );

	return true;
}

