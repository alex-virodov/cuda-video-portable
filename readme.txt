Demo code to try interoperation between OpenGL and CUDA.

Build (Windows):
Visual Studio 2010 project in 'win32' directory

Build (Linux):
Do 'make' in 'src'.




Directories:
  src/      - sources and headers
  bin_data/ - videos and .dll files
  win32/    - windows solution

Dependencies:
  - boost
  - wxwidgets
  - glew (unused)
  - opencv 
  
  Note: Put the appropriate dll's in 'bin_data' and select 'bin_data' as working directory.
        Alternatively, add dll directories to PATH
  