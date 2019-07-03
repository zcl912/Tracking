#include "stdafx.h"
#include <fstream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>

#include "Config.h"

Config::Config()
{

};

Config::Config(const Config &config)
	: /// Information
	sequence_name(config.sequence_name),

	sequence_path_thermal(config.sequence_path_thermal),//�����¼����thermal�Ⱥ���·��

	sequence_path(config.sequence_path),
	result_path(config.result_path),
	init_frame(config.init_frame),
	end_frame(config.end_frame),
	image_type(config.image_type),
	image_type_thermal(config.image_type_thermal),	//�����¼����thermal
	init_bbox(config.init_bbox),
	num_channel(config.num_channel),
	num_channel_thermal(config.num_channel_thermal),//�����¼����thermal
    patch_width(config.patch_width),
    patch_height(config.patch_height),

	bbox_width(config.bbox_width),
	bbox_height(config.bbox_height),

    scale_width(config.scale_width),
    scale_height(config.scale_height),
    search_radius(config.search_radius)
{
}


void Config::set(Config config)
{
  /// Information
  sequence_name = config.sequence_name;

  sequence_path_thermal = config.sequence_path_thermal;//�����¼����thermal�Ⱥ���·��

  sequence_path = config.sequence_path;
  result_path = config.result_path;  
  init_frame = config.init_frame;
  end_frame = config.end_frame;
  image_type = config.image_type;
  image_type_thermal = config.image_type_thermal;	//�����¼����thermal
  init_bbox.set(config.init_bbox);
  num_channel = config.num_channel;
  num_channel_thermal = config.num_channel_thermal;	//�����¼����thermal
  patch_width = config.patch_width;
  patch_height = config.patch_height;

  bbox_width = config.bbox_width; // ��ȡȫ�ֵ�
  bbox_height = config.bbox_height;

  scale_width = config.scale_width;
  scale_height = config.scale_height;
  search_radius = config.search_radius;
}