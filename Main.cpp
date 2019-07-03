#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <io.h>
#include <direct.h>
#include <Windows.h>

#include "opencv2/highgui/highgui.hpp"

#include "Config.h"
#include "Tracker.h"

#if _DEBUG
#pragma comment (lib, "opencv_core249d")
#pragma comment (lib, "opencv_highgui249d")
#pragma comment (lib, "opencv_imgproc249d")
#pragma comment (lib, "libmex")
#pragma comment (lib, "libmx")
#pragma comment (lib, "libmwlapack")
#pragma comment (lib, "libmwblas")
#else
#pragma comment (lib, "opencv_core249")
#pragma comment (lib, "opencv_highgui249")
#pragma comment (lib, "opencv_imgproc249")
#pragma comment (lib, "libmex")
#pragma comment (lib, "libmx")
#pragma comment (lib, "libmwlapack")
#pragma comment (lib, "libmwblas")
#endif

std::vector<char*> frame_files;
std::vector<char*> frame_files_thermal;

std::vector<char*> FindAllFiles(const char* folder); // read all subfolders
std::vector<char*> FindAllFiles(const char* folder, const char* suffix); // read all files with specific suffix

std::string sequence_name1;
int main(int argc, char* argv[])
{
	char temp[MAX_PATH];
	_getcwd(temp, MAX_PATH);
	std::string cur_path(temp);
	std::string data_path = "F:\\第一个工作\\第一个工作cvpr\\cvpr论文相关材料\\自己建立的工程VS2015 iet1\\ConsoleApplication1\\ConsoleApplication1\\Benchmark\\"; //RGB-D  -FOUR -ANALYSIS
  //Read Information
	std::vector<char*> sequences = FindAllFiles(data_path.c_str());

  Config config;

 //int i = 34;
  //for (int i = 0; i < 14; i++)
  //for (int i = 14; i < 26; i++)
 //for (int i = 26; i < 38; i++)
  //for (int i = 38; i < 50; i++)


   //for (int i = 0; i < 25; i++)
  // for (int i = 25; i < 50; i++)


   //for (int i = 0; i < 17; i++)
  //for (int i = 17; i < 34; i++)
  // for (int i = 34; i < 50; i++)

 // for (int i = 42; i < 50; i++)

 //for (int i = 0; i < 10; i++)
 // for (int i = 10; i < 20; i++)
 // for (int i = 20; i < 30; i++)
// for (int i = 30; i < 40; i++)
  for (int i = 40; i < 50; i++)

 // for (int i = 25; i < 50; i++)

//for (int i = 0; i < 28; i++)
// for (int i = 28; i < 50; i++)


 //for (int i = 28; i < 33; i++)
 // for (int i = 33; i < 38; i++)
//for (int i = 38; i < 43; i++)
//for (int i = 43; i < 46; i++)
 // for (int i = 46; i < 50; i++)

  //for (int i = 50; i < 60; i++)
  //for (int i = 60; i < 70; i++)
  //for (int i = 70; i < 80; i++)
  //for (int i = 80; i < 95; i++)
 //for (int i = 0; i < sequences.size(); i++)
  {
	//  config.sequence_path =/* cur_path + "\\" +*/ data_path + sequences[i] + "\\v\\";
	  config.sequence_path = data_path + sequences[i] + "\\v\\";
	  if (i == 32) frame_files = FindAllFiles(config.sequence_path.c_str(), "*.bmp"); // RGB-T
	  else frame_files = FindAllFiles(config.sequence_path.c_str(), "*.png");
	  //config.sequence_path = data_path + sequences[i] + "\\rgb\\";  //RGB-D
	  //frame_files = FindAllFiles(config.sequence_path.c_str(), "*.png");
	  cv::Mat frame = cv::imread(frame_files[0]);

	  /******新加入thermal热红外变量******/
	  config.sequence_path_thermal = data_path + sequences[i] + "\\i\\";
	  if (i == 32) frame_files_thermal = FindAllFiles(config.sequence_path_thermal.c_str(), "*.bmp");
	  else frame_files_thermal = FindAllFiles(config.sequence_path_thermal.c_str(), "*.png");
	  //config.sequence_path_thermal = data_path + sequences[i] + "\\depth\\";
	  //frame_files_thermal = FindAllFiles(config.sequence_path_thermal.c_str(), "*.png");
	  cv::Mat frame_thermal = cv::imread(frame_files_thermal[0]);
	  /******新加入thermal热红外变量******/

	  std::string gt_file = data_path + sequences[i] + "\\groundTruth_v.txt"; // x1, y1, x2, y2  RGB-T
	  //std::string gt_file = data_path + sequences[i] + "\\groundTruth_v.txt"; // x1, y1, w, h

	  std::fstream output(gt_file);
	  int x1, y1;
	  output >> config.init_bbox.x >> config.init_bbox.y >> x1 >> y1;   // RGB-T
	  //output >> config.init_bbox.x >> config.init_bbox.y >> config.init_bbox.w >> config.init_bbox.h;  //RGB-D

	  config.init_bbox.w = x1 - config.init_bbox.x;
	  config.init_bbox.h = y1 - config.init_bbox.y; //RGB-T

	  output.close();

	  //config.init_bbox.x += 8; config.init_bbox.y +=8; config.init_bbox.w -= 16; config.init_bbox.h -= 16;//NUS-PRO

	  config.sequence_name = sequences[i];
	  sequence_name1 = config.sequence_name;
	  config.init_frame = 1;

	  config.end_frame = frame_files.size();
	  config.num_channel = frame.channels();
	  config.num_channel_thermal = frame_thermal.channels();

	  config.init_bbox.x -= 1; // Matlab -> C++
	  config.init_bbox.y -= 1; // Matlab -> C++  

	  std::string path_tmp = "F://第一个工作//第五个工作 CVPR2018//tracker_macor_ver2(global)-RGBT - ver2 - gezi//results1";//
	  std::string path_bbox = path_tmp + "/bbox";
	  std::string path_iter_errs = path_tmp + "/iter_errs";
	  std::string path_Q = path_tmp + "/Q";
	  std::string path_query = path_tmp + "/query";
	  std::string path_R = path_tmp + "/R";
	  std::string path_S = path_tmp + "/S";  
	  //std::string path_weight = path_tmp + "/weight";
	  if (_access(path_tmp.c_str(), 0) == -1)	  _mkdir(path_tmp.c_str());
	  if (_access(path_bbox.c_str(), 0) == -1)	  _mkdir(path_bbox.c_str());
	  if (_access(path_iter_errs.c_str(), 0) == -1)	  _mkdir(path_iter_errs.c_str());
	  /*if (_access(path_Q.c_str(), 0) == -1)	  _mkdir(path_Q.c_str());
	  if (_access(path_R.c_str(), 0) == -1)	  _mkdir(path_R.c_str());
	  if (_access(path_query.c_str(), 0) == -1)	  _mkdir(path_query.c_str());
	  if (_access(path_S.c_str(), 0) == -1)	  _mkdir(path_S.c_str());*/

	  //config.result_path = std::string("../../results/tau/0.05");//results
	  config.result_path = path_tmp;//results


	  if (config.init_bbox.w < config.init_bbox.h)
	  {
		  config.scale_width = config.init_bbox.w / 32.0;
		  config.patch_width = std::round(config.init_bbox.w / (8.0*config.scale_width));
		  config.patch_height = std::round(config.init_bbox.h / (8.0*config.scale_width));
		  config.scale_height = config.init_bbox.h / (8.0*config.patch_height);

		  config.bbox_width = std::round(config.init_bbox.w / config.scale_width);
		  config.bbox_height = std::round(config.init_bbox.h / config.scale_width);
	  }
	  else
	  {
		  config.scale_height = config.init_bbox.h / 32.0;
		  config.patch_height = std::round(config.init_bbox.h / (8.0*config.scale_height));
		  config.patch_width = std::round(config.init_bbox.w / (8.0*config.scale_height));
		  config.scale_width = config.init_bbox.w / (8.0*config.patch_width);

		  config.bbox_height = std::round(config.init_bbox.h / config.scale_height);
		  config.bbox_width = std::round(config.init_bbox.w / config.scale_height);

	  }
	  config.search_radius
		  =1.0* sqrt(config.init_bbox.w*config.init_bbox.h / (config.scale_width*config.scale_height));

	  if (config.num_channel == 3)
		  config.image_type = cv::IMREAD_COLOR;
	  else
		  config.image_type = cv::IMREAD_GRAYSCALE;

	  if (config.num_channel_thermal == 3)//新加入的thermal
		  config.image_type_thermal = cv::IMREAD_COLOR;
	  else
		  config.image_type_thermal = cv::IMREAD_GRAYSCALE;

	  /// Tracking  
	  srand(0);
	  Tracker tracker(config);
	  tracker.run();
	  tracker.save(cur_path);
  }
  return 0;
}

std::vector<char*> FindAllFiles(const char* folderName)
{
	std::vector<char*> dirs;
	//存放初始目录的绝对路径，以\'\\'结尾
	char m_szInitDir[_MAX_PATH];
	memset(m_szInitDir, '\0', _MAX_PATH * sizeof(char));
	strcpy_s(m_szInitDir, folderName);
	//如果目录的最后一个字母不是\'\\',则在最后加上一个\'\\'
	int len = strlen(m_szInitDir);
	if (m_szInitDir[len - 1] != '\\')
		strcat_s(m_szInitDir, "\\");
	_chdir(m_szInitDir);
	//首先查找dir中符合要求的文件
	//long hFile;
	intptr_t hFile;
	_finddata_t fileinfo;
	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.name[1] == '\0' || fileinfo.name[2] == '\0')
				continue;
			char* filename = new char[_MAX_PATH];
			//strcpy_s(filename, _MAX_PATH, m_szInitDir);
			//strcat_s(filename, _MAX_PATH, fileinfo.name);
			strcpy_s(filename, _MAX_PATH, fileinfo.name);
			dirs.push_back(filename);

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	//sort(dirs.begin(), dirs.end());
	return dirs;
}

std::vector<char*> FindAllFiles(const char* folderName, const char* suffix)
{
	std::vector<char*> dirs;
	//存放初始目录的绝对路径，以\'\\'结尾
	char m_szInitDir[_MAX_PATH];
	memset(m_szInitDir, '\0', _MAX_PATH * sizeof(char));
	strcpy_s(m_szInitDir, folderName);
	//如果目录的最后一个字母不是\'\\',则在最后加上一个\'\\'
	int len = strlen(m_szInitDir);
	if (m_szInitDir[len - 1] != '\\')
		strcat_s(m_szInitDir, "\\");
	_chdir(m_szInitDir);
	//首先查找dir中符合要求的文件
	//long hFile;
	intptr_t hFile;
	_finddata_t fileinfo;
	if ((hFile = _findfirst(suffix, &fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果不是,则进行处理
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				char* filename = new char[_MAX_PATH];
				strcpy_s(filename, _MAX_PATH, m_szInitDir);
				strcat_s(filename, _MAX_PATH, fileinfo.name);
				dirs.push_back(filename);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	//sort(dirs.begin(), dirs.end());
	return dirs;
}
