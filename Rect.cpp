#include "stdafx.h"
#include <algorithm>
#include "Rect.h"

Rect::Rect() : x(0), y(0), w(0), h(0) {}
Rect::Rect(double x, double y, double w, double h) : x(x), y(y), w(w), h(h) {}
Rect::Rect(const Rect &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}

void Rect::set(double x, double y, double w, double h)
{
  this->x = x;
  this->y = y;
  this->w = w;
  this->h = h;
}
void Rect::set(const Rect &rect)
{
  x = rect.x;
  y = rect.y;
  w = rect.w;
  h = rect.h;
}

double Rect::area() const { return w*h; }
double Rect::overlap_ratio(const Rect &rect) const
{
  double x0 = std::max(x, rect.x);
  double x1 = std::min(x+w, rect.x+rect.w);
  double y0 = std::max(y, rect.y);
  double y1 = std::min(y+h, rect.y+rect.h);

  double interseted_area = (x1-x0)*(y1-y0);
  double unified_area = area() + rect.area() - interseted_area;

  return interseted_area/unified_area;
}

bool Rect::is_inside(const Rect &rect) const
{
  return (x >= rect.x) && (y >= rect.y) && (x+w <= rect.x+rect.w) && (y+h <= rect.y+rect.h);
}
