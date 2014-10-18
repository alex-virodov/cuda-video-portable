#ifndef __GLSCENE_H__
#define __GLSCENE_H__

class GLSceneImpl;

// ====================================================================
// Class to render the screen via OpenGL. Use pimpl to hide 
// implementation details, which are in flux.
class GLScene
{
public:
	GLScene();
	~GLScene();
	
	void resize(int width, int height);
	void render();

private:
	GLSceneImpl* impl;
};

#endif