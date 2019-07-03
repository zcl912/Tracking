#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include "Rect.h"

class Config
{
public:
  /// Information
  std::string sequence_name;
  std::string sequence_path_thermal;	//新加入的thermal热红外路径
  std::string sequence_path; 
  std::string result_path;
  int init_frame;
  int end_frame;  
  int image_type;
  int image_type_thermal;	//设置新加入的thermal
  Rect init_bbox;
  int num_channel; 
  int num_channel_thermal;	//设置新加入的thermal
  int patch_width;
  int patch_height;
  int search_radius;
  double scale_width;
  double scale_height;

  int bbox_width; //获取目标的宽、高
  int bbox_height;

  Config();
  Config(const Config &config);
  void set(Config config);
};

#endif