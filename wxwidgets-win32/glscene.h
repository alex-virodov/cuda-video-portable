#ifndef __GLSCENE_H__
#define __GLSCENE_H__

class GLSceneImpl;

class GLScene
{
public:
	GLScene();
	
	void resize(int width, int height);
	void render();
	void advance();

private:
	GLSceneImpl* impl;
};

#endif