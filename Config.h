#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include "Rect.h"

class Config
{
public:
  /// Information
  std::string sequence_name;
  std::string sequence_path_thermal;	//�¼����thermal�Ⱥ���·��
  std::string sequence_path; 
  std::string result_path;
  int init_frame;
  int end_frame;  
  int image_type;
  int image_type_thermal;	//�����¼����thermal
  Rect init_bbox;
  int num_channel; 
  int num_channel_thermal;	//�����¼����thermal
  int patch_width;
  int patch_height;
  int search_radius;
  double scale_width;
  double scale_height;

  int bbox_width; //��ȡĿ��Ŀ���
  int bbox_height;

  Config();
  Config(const Config &config);
  void set(Config config);
};

#endif