#include "stdafx.h"
#include <memory>
#include "GL/gl.h"
#include "GL/glu.h"

#include "glscene.h"
#include "me.h"
#include "video.h"

// Most render code is taken from Lesson 5 of NeHe OpenGL (3d version of hello world)
// http://nehe.gamedev.net/tutorial/3d_shapes/10035/

// CUDA interoperability
// http://3dgep.com/opengl-interoperability-with-cuda/


class GLSceneImpl
{
public:
	GLfloat	rtri;    // Angle For The Triangle ( NEW )
	GLfloat	rquad;   // Angle For The Quad ( NEW )
	
	GLuint  img_width;
	GLuint  img_height;

	GLuint  view_width;
	GLuint  view_height;

	static const int n_frames = 2;

	// TODO: Make texture management objects. Or use library.
	GLuint  tex_frames[n_frames]; // textures for frames
	GLuint  tex_result[1];        // textures for result of processing

	IMotionEstimation* me;
	std::unique_ptr<IVideoReader> video;

	GLSceneImpl() : rtri(0), rquad(200) 
	{
		tex_result[0] = 0;
		for (int i = 0; i < n_frames; i++) { tex_frames[i] = 0; }
	}

	void make_texture(GLuint tex_id, void* img_data = 0)
	{
		glBindTexture( GL_TEXTURE_2D, tex_id );

		// set basic parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,  GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,  GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		update_texture(tex_id, img_data);
	}

	void update_texture(GLuint tex_id, void* img_data)
	{
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data );
	}
};

void cuda_opengl_init(int n_frames, GLuint* tex_frames, GLuint tex_result);
void cuda_opengl_process();



GLScene::GLScene() : impl(new GLSceneImpl)
{
	glewInit();

	impl->video = std::unique_ptr<IVideoReader>(
		make_opencv_video_reader("clipcanvas_14348_H264_320x180.mp4"));

	assert(impl->n_frames == 2);

	if (impl->video->is_loaded())
	{
		impl->img_width  = impl->video->get_width();
		impl->img_height = impl->video->get_height();

		impl->me = make_me_cuda(impl->img_width, impl->img_height);

		glGenTextures( impl->n_frames, impl->tex_frames );
		glGenTextures(              1, impl->tex_result );

		// Make result texture
		impl->make_texture(impl->tex_result[0], /*img_data=*/0);

		// Make source frames
		for (int i = 0; i < impl->n_frames; i++) { impl->make_texture(impl->tex_frames[i]); }
	}
}

GLScene::~GLScene()
{
	glDeleteTextures(impl->n_frames, impl->tex_frames);
	glDeleteTextures(             1, impl->tex_result);
//	glDeleteBuffers (1, &impl->pbo);

	delete impl;
}

void GLScene::resize(int width, int height)
{
	// Initialization is also here
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearColor(0.1f, 0.1f, 0.1f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

    glClearColor(0.5f, 0.5f, 0.5f, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, (GLint)width, (GLint)height);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
//	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	// Record viewport size for later 3d perspective matrix computation
	impl->view_width  = width;
	impl->view_height = height;
}


void GLScene::render()
{
	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glLoadIdentity();									// Reset The Current Modelview Matrix
	glTranslatef(-1.5f,0.0f,-6.0f);						// Move Left 1.5 Units And Into The Screen 6.0
	glRotatef(impl->rtri,0.0f,1.0f,0.0f);						// Rotate The Triangle On The Y axis ( NEW )


	glLoadIdentity();									// Reset The Current Modelview Matrix
//	glTranslatef(0.0f,0.0f,-4.0f);						// Move Right 1.5 Units And Into The Screen 7.0
	//glRotatef(impl->rquad,1.0f,1.0f,1.0f);					// Rotate The Quad On The X axis ( NEW )

	glEnable(GL_TEXTURE_2D);							// Enable Texture Mapping 
	glColor3d(1,1,1);

	assert(impl->n_frames == 2);

	// Upload next video frame
	glBindTexture(GL_TEXTURE_2D, impl->tex_frames[0]);
	impl->update_texture(impl->tex_frames[0], impl->video->get_this_frame());
	
	impl->me->load_frame(0, impl->video->get_this_frame());

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 0.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 0.0f,  0.0f,  1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  0.0f,  1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, impl->tex_frames[1]);
	impl->update_texture(impl->tex_frames[0], impl->video->get_next_frame());

	impl->me->load_frame(1, impl->video->get_this_frame());

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 0.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  0.0f,  1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 0.0f,  0.0f,  1.0f);
	glEnd();

	impl->me->estimate();
	impl->me->store_result(impl->tex_result[0]);

	glBindTexture(GL_TEXTURE_2D, impl->tex_result[0]);
	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 0.0f,  0.0f,  1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  0.0f,  1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 0.0f, -1.0f,  1.0f);
	glEnd();

	// 3D Cube - just because why not :)
    glViewport(0, 0, impl->view_width/2, impl->view_height/2);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f,(GLfloat)impl->view_width/(GLfloat)impl->view_height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix
	glTranslatef(0.0f,0.0f,-5.0f);						// Move Right 1.5 Units And Into The Screen 7.0
	glRotatef(impl->rquad,1.0f,1.0f,1.0f);				// Rotate The Quad On The X axis ( NEW )

	glBindTexture(GL_TEXTURE_2D, impl->tex_frames[0]);        // Select Our Texture
	glBegin(GL_QUADS);
		// Front Face
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);  // Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);  // Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);  // Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);  // Top Left Of The Texture and Quad
		// Back Face
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);  // Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);  // Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);  // Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);  // Bottom Left Of The Texture and Quad
	glEnd();

	glBindTexture(GL_TEXTURE_2D, impl->tex_frames[1]);        // Select Our Texture
	glBegin(GL_QUADS);
		// Top Face
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);  // Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);  // Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);  // Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);  // Top Right Of The Texture and Quad
		// Bottom Face
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);  // Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, -1.0f);  // Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);  // Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);  // Bottom Right Of The Texture and Quad
	glEnd();

	glBindTexture(GL_TEXTURE_2D, impl->tex_result[0]);        // Select Our Texture
	glBegin(GL_QUADS);
		// Right face
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);  // Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);  // Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);  // Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);  // Bottom Left Of The Texture and Quad
		// Left Face
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);  // Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);  // Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);  // Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);  // Top Left Of The Texture and Quad
	glEnd();
}

void GLScene::advance()
{
	impl->rtri+=0.2f;											// Increase The Rotation Variable For The Triangle ( NEW )
	impl->rquad-=0.15f;										// Decrease The Rotation Variable For The Quad ( NEW )
}


