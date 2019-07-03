#ifndef RECT_H
#define RECT_H

class Rect
{
public:
  Rect();
  Rect(double x, double y, double w, double h);
  Rect(const Rect &bbox); 

  void set(double x, double y, double w, double h);
  void set(const Rect &bbox);

  bool is_inside(const Rect &bbox) const;
  double area() const;
  double overlap_ratio(const Rect &bbox) const;
  

public:
  double x;
  double y;
  double w;
  double h;  
};

#endif