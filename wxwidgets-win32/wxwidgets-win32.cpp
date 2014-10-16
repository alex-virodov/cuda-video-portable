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
	boost::thread* draw_thread;

public:
	virtual bool OnInit();

	void terminate_thread_blocking();
};

wxIMPLEMENT_APP_CONSOLE(MyApp);

// ====================================================================
// Window class
class MyFrame: public wxFrame
{
public:
	wxGLCanvas*    canvas;

	MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

	void onClose(wxCloseEvent& event);

private:
	wxDECLARE_EVENT_TABLE();
};

wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
	EVT_CLOSE(MyFrame::onClose)
wxEND_EVENT_TABLE()

MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
: wxFrame(NULL, wxID_ANY, title, pos, size)
{
	canvas = new wxGLCanvas(this, wxID_ANY, /*attribList=*/NULL);
}

void MyFrame::onClose(wxCloseEvent& event)
{
	wxGetApp().terminate_thread_blocking();
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

		cout << ":: still running" << endl;

		// TODO: is canvas->GetSize() thread-safe? probably not...
		scene.resize(canvas->GetSize().x, canvas->GetSize().y);
		scene.render();
		scene.advance();

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

bool MyApp::OnInit()
{
	MyFrame *frame = new MyFrame( "Hello World", wxPoint(50, 50), wxSize(450, 340) );
	

	frame->Show( true );

	draw_thread = new boost::thread(thread_func, frame->canvas);

	return true;
}


void MyApp::terminate_thread_blocking()
{
	if (draw_thread) {
		draw_thread->interrupt();
		draw_thread->join();
		delete draw_thread;
		draw_thread = 0;
	}
}
