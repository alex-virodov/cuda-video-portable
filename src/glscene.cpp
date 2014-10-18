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

// ====================================================================
// A simple OpenGL texture wrapper to handle initialization/deallocation
class GLTexture
{
	GLuint tex_id;
	GLuint tex_width;
	GLuint tex_height;
public:
	GLTexture(); 
	~GLTexture();

	void update_texture(GLuint width, GLuint height, void* img_data);

	GLuint get_id() { return tex_id; }
};

// ====================================================================
GLTexture::GLTexture()
{
	glGenTextures(1, &tex_id);

	glBindTexture( GL_TEXTURE_2D, tex_id );

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,  GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,  GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	tex_width = -1; tex_height = -1;
}

// ====================================================================
GLTexture::~GLTexture()
{
	glDeleteTextures(1, &tex_id);
}

// ====================================================================
void GLTexture::update_texture(GLuint width, GLuint height, void* img_data)
{
	if (tex_width == width && tex_height == height)
	{
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, img_data );
	} 
	else 
	{
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data );
		tex_width  = width;
		tex_height = height;
	}
}

// ====================================================================
// Actual implemenation of GLScene
class GLSceneImpl
{
public:
	GLfloat	cube_angle;
	
	GLuint  img_width;
	GLuint  img_height;

	GLuint  view_width;
	GLuint  view_height;

	// TODO: Use in alternating order. Now loading each frame twice.
	GLTexture frame1;
	GLTexture frame2;
	GLTexture result;

	std::auto_ptr<IMotionEstimation> me;
	std::auto_ptr<IVideoReader>      video;

	GLSceneImpl();

	void resize(int width, int height);
	void render();

	void render_3d_rect(float x_from, float y_from, 
		                float x_to,   float y_to,    
						bool both_cube_faces = false);
};

// ====================================================================
GLSceneImpl::GLSceneImpl() : cube_angle(200) 
{
	// Load video
	video = std::auto_ptr<IVideoReader>(
		make_opencv_video_reader("clipcanvas_14348_H264_320x180.mp4"));

	// Make motion estimation object
	if (video->is_loaded())
	{
		img_width  = video->get_width();
		img_height = video->get_height();

		me = std::auto_ptr<IMotionEstimation> (make_me_cuda(img_width, img_height));

		result.update_texture(img_width, img_height, /*data=*/0);
	}
}

// ====================================================================
void GLSceneImpl::resize(int width, int height)
{
	// Initialization is also here
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

    glClearColor(0.5f, 0.5f, 0.5f, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

	// Record viewport size for later 3d perspective matrix computation
	view_width  = width;
	view_height = height;
}

// ====================================================================
void GLSceneImpl::render()
{
	glViewport(0, 0, (GLint)view_width, (GLint)view_height);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glLoadIdentity();									// Reset The Current Modelview Matrix
	glTranslatef(-1.5f,0.0f,-6.0f);						// Move Left 1.5 Units And Into The Screen 6.0


	glLoadIdentity();									// Reset The Current Modelview Matrix
//	glTranslatef(0.0f,0.0f,-4.0f);						// Move Right 1.5 Units And Into The Screen 7.0
	//glRotatef(rquad,1.0f,1.0f,1.0f);					// Rotate The Quad On The X axis ( NEW )

	glEnable(GL_TEXTURE_2D);							// Enable Texture Mapping 
	glColor3d(1,1,1);

	// Upload next video frame
	glBindTexture(GL_TEXTURE_2D, frame1.get_id());
	frame1.update_texture(img_width, img_height, video->get_this_frame());
	
	me->load_frame(0, video->get_this_frame());

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 0.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 0.0f,  0.0f,  1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  0.0f,  1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, frame2.get_id());
	frame2.update_texture(img_width, img_height, video->get_next_frame());

	me->load_frame(1, video->get_this_frame());

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 0.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  0.0f,  1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 0.0f,  0.0f,  1.0f);
	glEnd();

	// Call CUDA processing
	me->estimate();
	me->store_result(result.get_id());

	glBindTexture(GL_TEXTURE_2D, result.get_id());
	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 0.0f,  0.0f,  1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  0.0f,  1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 0.0f, -1.0f,  1.0f);
	glEnd();

	// 3D Cube - just because why not :)
    glViewport(0, 0, view_width/2, view_height/2);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f,(GLfloat)view_width/(GLfloat)view_height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix
	glTranslatef(0.0f,0.0f,-5.0f);						// Move Right 1.5 Units And Into The Screen 7.0
	glRotatef(cube_angle,1.0f,1.0f,1.0f);				// Rotate The Quad On The X axis ( NEW )

	glBindTexture(GL_TEXTURE_2D, frame1.get_id());        // Select Our Texture
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

	glBindTexture(GL_TEXTURE_2D, frame2.get_id());        // Select Our Texture
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

	glBindTexture(GL_TEXTURE_2D, result.get_id());        // Select Our Texture
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

	cube_angle -= 0.15f;
}

// ====================================================================
// GLScene proxy functions
GLScene::GLScene()
{
	impl = new GLSceneImpl();
}

GLScene::~GLScene()
{
	delete impl;
}

void GLScene::resize(int width, int height)
{
	impl->resize(width, height);
}

void GLScene::render()
{
	impl->render();
}

