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

	static const int n_frames = 2;

	// TODO: Make texture management objects. Or use library.
	GLuint  tex_frames[n_frames]; // textures for frames
	GLuint  tex_result[1];        // textures for result of processing

	IMotionEstimation* me;

	GLSceneImpl() : rtri(0), rquad(200) 
	{
		tex_result[0] = 0;
		for (int i = 0; i < n_frames; i++) { tex_frames[i] = 0; }
	}

	void make_texture(GLuint tex_id, void* img_data)
	{
		glBindTexture( GL_TEXTURE_2D, tex_id );

		// set basic parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data );
	}
};

void cuda_opengl_init(int n_frames, GLuint* tex_frames, GLuint tex_result);
void cuda_opengl_process();



GLScene::GLScene() : impl(new GLSceneImpl)
{
	glewInit();

	std::unique_ptr<IVideoReader> frame1 /*=*/ (make_opencv_image_reader("ball_frame1.bmp"));
	std::unique_ptr<IVideoReader> frame2 /*=*/ (make_opencv_image_reader("ball_frame2.bmp"));

	assert(impl->n_frames == 2);

	if (frame1->is_loaded() && frame2->is_loaded())
	{
		assert(frame1->get_width () == frame2->get_width ());
		assert(frame1->get_height() == frame2->get_height());

		impl->img_width  = frame1->get_width();
		impl->img_height = frame1->get_height();

		impl->me = make_me_cuda(impl->img_width, impl->img_height);

		glGenTextures( impl->n_frames, impl->tex_frames );
		glGenTextures(              1, impl->tex_result );

		// Make result texture
		impl->make_texture(impl->tex_result[0], /*img_data=*/0);

		// Make source frames
		for (int i = 0; i < impl->n_frames; i++) 
		{
			// TODO: work with lists if more than two frames are expected
			void* img_data = (i == 0 ? frame1->get_next_frame() : frame2->get_next_frame());

			impl->make_texture(impl->tex_frames[i], img_data);
		}

		impl->me->load_frame(0, frame1->get_next_frame());
		impl->me->load_frame(1, frame2->get_next_frame());
		impl->me->estimate();
		impl->me->store_result(impl->tex_result[0]);





#if 0
		cuda_opengl_init(impl->n_frames, impl->tex_frames, impl->tex_result[0]);



//		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image->width, image->height, GL_RGBA, GL_UNSIGNED_BYTE, image->data);


		// TODO: Check that we have the PBO extension
		glGenBuffers( 1, &impl->pbo );
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, impl->pbo );
		glBufferData( GL_PIXEL_UNPACK_BUFFER, image->width*image->height*4, image->data, GL_STREAM_DRAW );
 

		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, impl->pbo);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image->width, image->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);


		// Unbind the texture
		glBindBuffer   ( GL_PIXEL_UNPACK_BUFFER, 0 );
		glBindTexture  ( GL_TEXTURE_2D, 0 );

		cuda_opengl_init(impl->texture, impl->pbo);

		// Copy from PBO to texture
//		cuda_opengl_process();
#endif
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
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, (GLint)width, (GLint)height);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
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
	glTranslatef(0.0f,0.0f,-7.0f);						// Move Right 1.5 Units And Into The Screen 7.0
	glRotatef(impl->rquad,1.0f,1.0f,1.0f);					// Rotate The Quad On The X axis ( NEW )

	glEnable(GL_TEXTURE_2D);							// Enable Texture Mapping 
	glColor3d(1,1,1);

	assert(impl->n_frames == 2);

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


