// Author(s): Carst Tankink and Ali Deniz Aladagli
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file glcanvas.cpp
/// \brief Implementation of OpenGL rendering canvas.

#include "wx.hpp" // precompiled headers

#include "glcanvas.h"
#include <wx/dcclient.h>
#include "icons/zoom_cursor.xpm"
#include "icons/zoom_cursor_mask.xpm"
#include "icons/pan_cursor.xpm"
#include "icons/pan_cursor_mask.xpm"
#include "icons/rotate_cursor.xpm"
#include "icons/rotate_cursor_mask.xpm"
#include <iostream>

#include "ids.h"

#ifdef __APPLE__
  #include <OpenGL/glu.h>
#else
  #include <GL/glu.h>
#endif
using namespace IDS;
BEGIN_EVENT_TABLE(GLCanvas, wxGLCanvas)
  EVT_PAINT(GLCanvas::onPaint)
  EVT_SIZE(GLCanvas::onSize)

  EVT_ENTER_WINDOW(GLCanvas::onMouseEnter)
  EVT_KILL_FOCUS(GLCanvas::onMouseLeave)
  EVT_LEFT_DOWN(GLCanvas::onMouseLftDown)
  EVT_RIGHT_DOWN(GLCanvas::onMouseRgtDown)
  EVT_LEFT_UP(GLCanvas::onMouseLftUp)
  EVT_RIGHT_UP(GLCanvas::onMouseRgtUp)
  EVT_MOTION(GLCanvas::onMouseMove)
  EVT_MIDDLE_DOWN(GLCanvas::onMouseMidDown)
  EVT_MIDDLE_UP(GLCanvas::onMouseMidUp)
  EVT_MOUSEWHEEL(GLCanvas::onMouseWhl)
  EVT_LEFT_DCLICK(GLCanvas::onMouseDblClck)

END_EVENT_TABLE()

GLCanvas::GLCanvas(LTSGraph* app, wxWindow* parent,
                   const wxSize &size, int* attribList)
/* Deviation required to eliminate flickering of states */
#if wxCHECK_VERSION(2, 9, 0)
	: wxGLCanvas(parent, wxID_ANY, NULL /* attribs */,
             wxDefaultPosition, wxDefaultSize,
             wxFULL_REPAINT_ON_RESIZE | wxSUNKEN_BORDER)
# else
	: wxGLCanvas(parent, wxID_ANY, wxDefaultPosition, size, wxSUNKEN_BORDER  | wxFULL_REPAINT_ON_RESIZE,
               wxEmptyString, attribList)
#endif
{
  owner = app;
  dispSystem = false;
  currentTool = myID_ZOOM;
  usingTool = false;
  calcRot = false;
  drawIn3D = false;
  lookX = 0;
  lookY = 0;
  lookZ = 0;
  rotX = 0;
  rotY = 0;
  scaleFactor = 1.0;  
  double someMatrix[] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  for ( int i = 0; i < 16; i++)
    currentModelviewMatrix[i] = someMatrix[i];
}

GLCanvas::~GLCanvas()
{
}

void GLCanvas::initialize()
{
	/* Following line causes segfault in wxWidgets >= 2.9 */
#if wxCHECK_VERSION(2, 9, 0) 	
#else
	visualizer->initFontRenderer();
#endif	
}

void GLCanvas::setVisualizer(Visualizer *vis)
{
  visualizer = vis;
}

void GLCanvas::render2D()
{

    // Reset User perspective
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    double wwidth, wheight, wdepth;
    getSize(wwidth, wheight, wdepth);

    int width, height;
    GetClientSize( &width, &height);
    // Cast to GLdouble for smooth transitions
    GLdouble aspect = (GLdouble)getAspectRatio();

    glViewport(0, 0, width, height);
    //  Define the 2D orthographic projection matrix
    if (aspect > 1)
    {
      // width > height
      gluOrtho2D(aspect*(-1), aspect, -1.0, 1.0);
    }
    else
    {
      // height >= width
      gluOrtho2D(-1.0, 1.0, (1/aspect)*(-1), (1/aspect));
    }

    // Draw
    if (visualizer)
    {
      visualizer->visualize(wwidth, wheight, getPixelSize(), false, false);
    }
}


void GLCanvas::render3D()
{

    // Set up viewing volume
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    double wwidth, wheight, wdepth;
    getSize(wwidth, wheight, wdepth);

    int width, height;
    GetClientSize( &width, &height);
	glViewport(0, 0, width, height);

    // Cast to GLdouble for smooth transitions
    GLdouble aspect = (GLdouble)getAspectRatio();

	double rad = visualizer->getRadius() * getPixelSize() ;
	maxDepth = std::max(std::max((wdepth - 2 * rad), (wheight - 2 * rad)), (wwidth - 2 * rad));
	gluPerspective(45.0f, aspect, 0.1f, 2 * (lookZ + maxDepth + 0.1f));

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if(calcRot)
	{
	  double dumtrx[16];
	  double rotAngle = sqrt(rotX * rotX + rotY * rotY);

	  Utils::genRotArbAxs(rotAngle, rotX, rotY, 0, dumtrx);
	  double dumtrx2[16];
	  Utils::MultGLMatrices(dumtrx, currentModelviewMatrix, dumtrx2);
	  for ( int i = 0; i < 12; i++)
		currentModelviewMatrix[i] = dumtrx2[i];
	  calcRot = false;
	  normalizeMatrix();
	  rotX = 0;
	  rotY = 0;
	}
	currentModelviewMatrix[12] = -lookX;
	currentModelviewMatrix[13] = -lookY;
	currentModelviewMatrix[14] = -lookZ - 0.1f - maxDepth / 2;
	currentModelviewMatrix[15] = 1;
	glLoadMatrixd(currentModelviewMatrix);

	if (visualizer)
	{
	  // Draw
	  visualizer->visualize(wwidth, wheight, getPixelSize(), false, true);\

	  // Draw coordinate system (if enabled)
	  if(dispSystem){
	    glViewport(0, 0, std::max(height, width) / 6, std::max(height, width) / 6);
	    visualizer->drawCoorSystem();
	  }
	}
};

void GLCanvas::display()
{

  wxPaintDC dc(this);
  GLContext *glcontext = owner->getMainFrame()->getGLContext( this );

  if(!drawIn3D){
	  glcontext->set2DContext();
	  render2D();
  } else {
	  glcontext->set3DContext();
	  render3D();
  }

  /* Ensure all GL commands are processed */
  glFinish();

  /* Display if any errors have occurred */
  GLenum error = glGetError();
  if (error != GL_NO_ERROR)
  {
    std::cerr << "OpenGL error: " << gluErrorString(error) << std::endl;
  }

  /* Swap render buffer */
  SwapBuffers();

  return;
}

void GLCanvas::onPaint(wxPaintEvent& /*event*/)
{
  display();
}

void GLCanvas::onSize(wxSizeEvent& )
{
	recalcPixelSize();
	recalcAspectRatio();
}

void GLCanvas::getSize(
  double &width,
  double &height,
  double &depth)
/* (Based on method of the same name in Diagraphica's GLcanvas, the following
 * description derrives from that tool)
 * Returns viewport width and height in world coordinates.
 *
 * Before scaling, the viewport is set up such that the shortest side has length
 * 2 in world coordinates. Let ratio = width/height be the aspect ration of the
 * window and keep in mind that:
 *
 *  Size(viewport)   = Size(world)*scaleFactor
 *  So, Size(world)  = Size(viewport)/ScaleFactor
 *  So, Size(world)  = 2/ScaleFactor
 * There are 2 cases:
 * (1) If aspect > 1, the viewport is wider than tall
 *     So, the starting height was 2:
 *     world width      = ( aspect*2 ) / scaleFactor
 *     world height     = 2 / scaleFactor
 * (2) If aspect <= 1, the viewport is taller than wide
 *     So, the starting width was 2:
 *     world width      = 2 / scaleFactor;
 *     world height     = ( aspect*2 ) / scaleFactor
 * Depth is just added for ease of use in 3D spaces.
 */
{
  if (getAspectRatio() > 1)
  {
    // width > height, so starting height was 2
    width = (getAspectRatio() * 2.0) / (double) scaleFactor;
    height = 2.0 / (double) scaleFactor;
  }
  else
  {
    // height >= width, so starting width was 2
    width = 2.0 / (double) scaleFactor;
    height = ((1/getAspectRatio()) * 2.0) / (double) scaleFactor;
  }
  depth = (width + height) / 2;
}

void GLCanvas::recalcPixelSize()
// TODO: Comment
{
  double result = 0.0;

  int widthPixels;
  int heightPixels;

  /* Get size of current canvas*/
  this->GetSize(
    &widthPixels,
    &heightPixels);

  double widthWorld;
  double heightWorld;
  double depthWorld;

  getSize(widthWorld, heightWorld, depthWorld);

  result = widthWorld * (1 / static_cast<double>(widthPixels));

  pixelSize = result;
}

double GLCanvas::getPixelSize()
{
	return pixelSize;
}

void GLCanvas::onMouseEnter(wxMouseEvent& /* event */)
{
  this->SetFocus();
}

void GLCanvas::onMouseLeave(wxFocusEvent& /* event */) 
{
  owner->deselect();
}

void GLCanvas::onMouseLftDown(wxMouseEvent& event)
{
  oldX = event.GetX();
  oldY = event.GetY();
  if(drawIn3D)
  {
    if (pickObjects3d(oldX, oldY, event))
    {
      owner->dragObject();
      usingTool = false;
    }
    else
      usingTool = true;
  }
  else
  {
    pickObjects(oldX, oldY, event);
    owner->dragObject();
  }
}

void GLCanvas::onMouseLftUp(wxMouseEvent& /* evt */)
{
  usingTool = false;
  setMouseCursor(myID_NONE);
  owner->stopDrag();
}

void GLCanvas::onMouseRgtDown(wxMouseEvent& event)
{
  oldX = event.GetX();
  oldY = event.GetY();

  if(drawIn3D)
  {
    if (pickObjects3d(oldX, oldY, event))
    {
      owner->lockObject();
      usingTool = false;
    }
    else
      usingTool = true;  
  }
  else
  {
    pickObjects(oldX, oldY, event);
    owner->lockObject();
  }
}

void GLCanvas::onMouseRgtUp(wxMouseEvent& /*evt */)
{
  setMouseCursor(myID_NONE);
  usingTool = false;
}
void GLCanvas::onMouseDblClck(wxMouseEvent& /*evt */)
{
//to be assigned
}
void GLCanvas::onMouseWhl(wxMouseEvent& event)
{
  if(drawIn3D)
  {
    lookZ -= double(event.GetWheelRotation())/(2400.0f);
  }
  wxPaintEvent evt = wxPaintEvent();
  this->GetEventHandler()->ProcessEvent(evt);
}
void GLCanvas::onMouseMidDown(wxMouseEvent& event)
{
  if(drawIn3D)
  {
    oldX = event.GetX();
    oldY = event.GetY();
  }
}
void GLCanvas::onMouseMidUp(wxMouseEvent& /*evt */)
{
  if(drawIn3D)
  {
    setMouseCursor(myID_NONE);
  }
}

void GLCanvas::onMouseMove(wxMouseEvent& event)
{
  if(drawIn3D)
  {
    if(event.Dragging() && (event.LeftIsDown() || event.MiddleIsDown() || event.RightIsDown()))
    {
    int x, y;
    GetPosition(&x, &y);
    int newX = static_cast<int>(event.GetX());
    int newY = static_cast<int>(event.GetY());
    int width, height;
    GetClientSize(&width, &height); 
    if (event.LeftIsDown() && !event.RightIsDown())
    {
      if(!usingTool)
      {
        double invect[] = {newX - oldX, oldY - newY, 0, 1};
        
        owner->moveObject(invect);
      }
      else
      {
        switch(currentTool)
        {
          case myID_PAN:
            lookX += 0.01f * (oldX - newX);
            lookY += -0.01f * (oldY - newY);
            break;
          case myID_ZOOM:
            lookZ += -0.01f * (oldY - newY);
            break;
          case myID_ROTATE:
            rotX = 0.5f * (oldX - newX);
            rotY = -0.5f * (oldY - newY);
            calcRot = true;
          default:
            break;
        }
        setMouseCursor(currentTool);
      }
    }
    else if ((event.MiddleIsDown() || (event.RightIsDown() && usingTool)) && !event.LeftIsDown())
    {
      rotX = 0.5f * (oldX - newX);
      rotY = -0.5f * (oldY - newY);
      calcRot = true;
      setMouseCursor(myID_ROTATE);
    }
    if ((x < newX) && (newX < x + width)) {
      oldX = newX;
    }
    if ((y < newY) && (newY < y + height)) {
      oldY = newY;
    }

    wxPaintEvent ev = wxPaintEvent();
    GetEventHandler()->ProcessEvent( ev );
    }
  }
  else
  {
    if(event.Dragging() && event.LeftIsDown())
    {
      int width, height;
      int x, y;
      GetPosition(&x, &y);
      GetClientSize(&width, &height);

      int newX = static_cast<int>(event.GetX());
      int newY = static_cast<int>(event.GetY());

      double diffX = static_cast<double>(newX - oldX) / static_cast<double>(width) * 2000;
      double diffY = static_cast<double>(oldY - newY) / static_cast<double>(height) * 2000;

      if ((x < newX) && (newX < x + width)) {
        oldX = newX;
      }
      if ((y < newY) && (newY < y + height)) {
        oldY = newY;
      }

      owner->moveObject(diffX, diffY);

      wxPaintEvent ev = wxPaintEvent();
      this->GetEventHandler()->ProcessEvent( ev );

    }
  }
}


bool GLCanvas::pickObjects3d(int x, int y, wxMouseEvent const& e)
{
  owner->deselect();

  GLContext *glcontext = owner->getMainFrame()->getGLContext( this );

  if(glcontext)
  {
    GLuint selectBuf[512];
    GLint  hits = 0;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glSelectBuffer(512, selectBuf);
    // Swith to selection mode
    (void) glRenderMode(GL_SELECT);

    glInitNames();

    // Create new projection transformation
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    // Create picking region near cursor location
    gluPickMatrix((GLdouble) x,
                  (GLdouble)  viewport[3] - y,
                  1.0,
                  1.0,
                  viewport);

    // Get current size of canvas
    int width,height,depth;
    GetClientSize(&width,&height);
    depth = (width + height) / 2;

    GLdouble aspect = (GLdouble)getAspectRatio();

    double wwidth, wheight, wdepth;
    getSize(wwidth, wheight, wdepth);

  double rad = visualizer->getRadius() * getPixelSize() ;

  maxDepth = std::max(std::max((wdepth - 2 * rad), (wheight - 2 * rad)), (wwidth - 2 * rad));

  gluPerspective(45.0f, aspect, 0.1f, 2 * (lookZ + maxDepth + 0.1f));

    glMatrixMode( GL_MODELVIEW);

    visualizer->visualize(wwidth, wheight, getPixelSize(), true, true);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    

    hits = glRenderMode(GL_RENDER);

    processHits(hits, selectBuf, e);
  return hits > 0;
  }
  return false;
}

void GLCanvas::pickObjects(int x, int y, wxMouseEvent const& e)
{
  owner->deselect();

  GLContext *glcontext = owner->getMainFrame()->getGLContext( this );

  if(glcontext)
  {
    GLuint selectBuf[512];
    GLint  hits = 0;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glSelectBuffer(512, selectBuf);
    // Swith to selection mode
    (void) glRenderMode(GL_SELECT);

    glInitNames();

    // Create new projection transformation
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    // Create picking region near cursor location
    gluPickMatrix((GLdouble) x,
                  (GLdouble)  viewport[3] - y,
                  5.0,
                  5.0,
                  viewport);

    // Get current size of canvas
    int width,height;
    GetClientSize(&width,&height);

    GLdouble aspect = (GLdouble)width / (GLdouble)height;

    if (aspect > 1)
    {
      // width > height
      gluOrtho2D(aspect * (-1), aspect, -1, 1);
    }
    else
    {
      // height >= width
      gluOrtho2D(-1, 1, -1/aspect, (1/aspect));
                               // calculate rotations etc.
    }

    glMatrixMode( GL_MODELVIEW);
    double wwidth, wheight, wdepth;
    getSize(wwidth, wheight, wdepth);

    visualizer->visualize(wwidth, wheight, getPixelSize(), true, false);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glFlush();

    hits = glRenderMode(GL_RENDER);

    processHits(hits, selectBuf, e);
  }
}

void GLCanvas::processHits(const GLint hits, GLuint *buffer, wxMouseEvent const& e)
{
  // This method selects the object clicked.
  //
  // The buffer content per hit is encoded as follows:
  //  buffer[0]: The number of names on the name stack at the moment of the hit
  //  buffer[1]: The minimal depth of the object hit
  //  buffer[2]: The maximal depth of the object. We are certainly not
  //             interested in this.
  //  buffer[3]: The first identifier of the object picked.
  // (buffer[4]: The second identifier of the object picked.)
  int selectedObject[3]; // Identifier for the object picked
  selectedObject[0] = -1; // None selected
  GLuint names; // The number of names on the stack.
  bool stateSelected = false; // Whether we've hit a state
  bool transSelected = false; // Whether we've hit a transition

  // Choose the objects hit, and store it. Give preference to states, then
  // to transition handles, then to labels
  using namespace IDS;
  for(GLint j = 0; j < hits; ++j) {
    names = *buffer;
    ++buffer; // Buffer points to minimal z value of hit.
    ++buffer; // Buffer points to maximal z value of hit.
    ++buffer; // Buffer points to first name on stack
    GLuint objType = *buffer;

    for(GLuint k = 0; k < names; ++k) {
      if(!(stateSelected || transSelected) || objType == STATE) {
        selectedObject[k] = *buffer;
      }
      ++buffer;
  }

    stateSelected = objType == STATE;
    transSelected = objType == TRANSITION || objType == SELF_LOOP;
  }


  switch(selectedObject[0])
  {
    case TRANSITION: {
      owner->selectTransition(selectedObject[1], selectedObject[2]);
      break;
    }
    case IDS::SELF_LOOP:
    {
      owner->selectSelfLoop(selectedObject[1], selectedObject[2]);
      break;
    }
    case IDS::STATE:
    {
      if(!e.CmdDown()) {
        owner->selectState(selectedObject[1]);
      }
      if (e.Button(wxMOUSE_BTN_LEFT)) {
        owner->colourState(selectedObject[1]);
      }
      else {
        owner->uncolourState(selectedObject[1]);
      }
      break;
    }
    case IDS::LABEL:
    {
      owner->selectLabel(selectedObject[1], selectedObject[2]);
      break;
    }
    case IDS::SELF_LABEL:
    {
      owner->selectSelfLabel(selectedObject[1], selectedObject[2]);
      break;
    }
    default: break;
  }

  buffer = NULL;
}

void GLCanvas::recalcAspectRatio()
{
  int width, height;

  GetClientSize(&width, &height);

  aspectRatio = static_cast<double>(width) / static_cast<double>(height);
}



double GLCanvas::getAspectRatio()
{
  return aspectRatio;
}

double GLCanvas::getMaxDepth() const
{
  return maxDepth;
}

void GLCanvas::getMdlvwMtrx(double * mtrx)
{
  for (int i = 0; i < 16; i++)
    mtrx[i] = currentModelviewMatrix[i];
}

void GLCanvas::getCamPos(double & x, double & y, double & z)
{
  x = lookX;
  y = lookY;
  z = lookZ + 0.1f + maxDepth / 2;
}

void GLCanvas::ResetAll()
{
  ResetRot();
  ResetPan();
}

void GLCanvas::ResetRot()
{
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glGetDoublev(GL_MODELVIEW_MATRIX, currentModelviewMatrix);
  glPopMatrix();
  currentModelviewMatrix[12] = -lookX;
  currentModelviewMatrix[13] = -lookY;
  currentModelviewMatrix[14] = -lookZ - 0.1f - maxDepth / 2;
  currentModelviewMatrix[15] = 1; 
}

void GLCanvas::ResetPan()
{
  lookX = 0;
  lookY = 0;
  lookZ = 0;
}
void GLCanvas::setMode(int tool)
{
  currentTool = tool;
}

void GLCanvas::showSystem()
{
  dispSystem = !dispSystem;
}

void GLCanvas::normalizeMatrix()
{
  double x, y, z, norm;
  bool xAx = currentModelviewMatrix[1] < 0;
  Utils::Vect yAx;
  x = currentModelviewMatrix[8];
  y = currentModelviewMatrix[9];
  z = currentModelviewMatrix[10];
  yAx.x = currentModelviewMatrix[4];
  yAx.y = currentModelviewMatrix[5];
  yAx.z = currentModelviewMatrix[6];
  norm = sqrt(x * x + y * y + z * z);
  x = x / norm;
  y = y / norm;
  z = z / norm;

  double tanXZ = atan2(x, z) * 180.0f / M_PI;
  double angYxz = atan2(y, sqrt(x * x + z * z)) * 180.0f / M_PI;
  glPushMatrix();
  glLoadIdentity();
  glRotated(tanXZ, 0.0, 1.0, 0.0);
  glRotated(angYxz, -1.0, 0.0, 0.0);
  glGetDoublev(GL_MODELVIEW_MATRIX, currentModelviewMatrix);
  Utils::Vect normal;
  normal.x = currentModelviewMatrix[8];
  normal.y = currentModelviewMatrix[9];
  normal.z = currentModelviewMatrix[10];
  normal = normal / Utils::vecLength(normal);
  Utils::Vect projection = yAx - (Utils::dotProd(yAx, normal)) * normal;
  yAx.x = currentModelviewMatrix[4];
  yAx.y = currentModelviewMatrix[5];
  yAx.z = currentModelviewMatrix[6];
  double angleRot = Utils::angDiff(yAx, projection) * 180.0f / M_PI;
  if(xAx)
    glRotated(-angleRot, 0.0, 0.0, 1.0);
  else
    glRotated(angleRot, 0.0, 0.0, 1.0);
  glGetDoublev(GL_MODELVIEW_MATRIX, currentModelviewMatrix);
  glPopMatrix();
}

void GLCanvas::setMouseCursor(int theTool) 
{
  wxCursor cursor;
  wxImage img;
  bool ok = true;
  switch (theTool) 
  {
    case myID_NONE:
      cursor = wxNullCursor;
      break;
    case myID_ZOOM:
      img = wxImage(zoom_cursor);
      img.SetMaskFromImage(wxImage(zoom_cursor_mask),255,0,0);
      cursor = wxCursor(img);
      break;
    case myID_PAN:
      img = wxImage(pan_cursor);
      img.SetMaskFromImage(wxImage(pan_cursor_mask),255,0,0);
      cursor = wxCursor(img);
      break;
    case myID_ROTATE:
      img = wxImage(rotate_cursor);
      img.SetMaskFromImage(wxImage(rotate_cursor_mask),255,0,0);
      cursor = wxCursor(img);
      break;
    default:
      ok = false;
      break;
  }
  if (ok)
    SetCursor(cursor);
}

void GLCanvas::changeDrawMode()
{
  drawIn3D = !drawIn3D;
  initialize();
}

bool GLCanvas::get3D()
{
  return drawIn3D;
}
