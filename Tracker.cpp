#include "stdafx.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/LU>
#include <chrono>

#include<cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tracker.h"

//定义外部变量——在其它地方已被定义，在这里要使用这个变量
extern std::vector<char*> frame_files;
extern std::vector<char*> frame_files_thermal;
extern std::string sequence_name1;


Tracker::Tracker(Config &config) :
	/// Constant Setting
	SEQUENCE_NAME(config.sequence_name),
	SEQUENCE_PATH(config.sequence_path),
	/*****************自己加的热红外*******************/
	SEQUENCE_PATH_THERMAL(config.sequence_path_thermal),
	/*****************自己加的热红外*******************/
	RESULT_PATH(config.result_path),
	IMAGE_TYPE(config.image_type),
	IMAGE_TYPE_THERMAL(config.image_type_thermal),//新加入的thermal
	INIT_FRAME(config.init_frame),
	END_FRAME(config.end_frame),
	NUM_FRAME(END_FRAME - INIT_FRAME + 1),
	//NUM_FRAME(1),
	NUM_CHANNEL(config.num_channel + 1),
	PATCH_DIM(NUM_CHANNEL*CHANNEL_DIM),
	OBJECT_DIM(NUM_PATCH*PATCH_DIM),
	BBOX_DIM(1 * PATCH_DIM), // 整个目标的维度，即patch的维度（32维度）,目标变成整个bounding box
	INIT_BOX(config.init_bbox)
	{
		/// Variable Setting
		patch_w = config.patch_width;
		patch_h = config.patch_height;
		scale_w = config.scale_width;
		scale_h = config.scale_height;
		search_r = config.search_radius;

		bbox_w = config.bbox_width; //全局的
		bbox_h = config.bbox_height;

		is_scale = false;

		char image_name[100];

		image = cv::imread(frame_files[0], IMAGE_TYPE);
		cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

		/***********自己添加的表示热红外图像********/
		image_thermal = cv::imread(frame_files_thermal[0], IMAGE_TYPE_THERMAL);
		if (SEQUENCE_NAME == "occBike")
		{
			cv::copyMakeBorder(image_thermal, image_thermal, 0, 8, 0, 16, cv::BORDER_CONSTANT, cv::Scalar());
		}
		cv::resize(image_thermal, image_thermal, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image_thermal, image_thermal, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());
		/***********自己添加的表示热红外图像********/

		border_bbox.set(0, 0, image.cols - 1, image.rows - 1);
		image_bbox.set(patch_w, patch_h, border_bbox.w - 2 * patch_w, border_bbox.h - 2 * patch_h);

		object_bbox.x = std::round(config.init_bbox.x / scale_w) + patch_w;
		object_bbox.y = std::round(config.init_bbox.y / scale_h) + patch_h;
		object_bbox.w = std::round(config.init_bbox.w / scale_w);
		object_bbox.h = std::round(config.init_bbox.h / scale_h);

		if (object_bbox.x < 0)
			object_bbox.x = 0;
		if (object_bbox.x + object_bbox.w > image_bbox.x + image_bbox.w)
			object_bbox.x = image_bbox.x + image_bbox.w - object_bbox.w;
		if (object_bbox.y < 0)
			object_bbox.y = 0;
		if (object_bbox.y + object_bbox.h > image_bbox.y + image_bbox.h)
			object_bbox.y = image_bbox.y + image_bbox.h - object_bbox.h;

		result_box.resize(NUM_FRAME, Rect());

		/********自己添加的 结果包围盒我没有用同一个*********/
		result_box_thermal.resize(NUM_FRAME, Rect());//是否可以考虑使用同一个？
		/********自己添加的 结果包围盒我没有用同一个*********/
}

Tracker::~Tracker() {};

void Tracker::display_result(int t, std::string window_name, std::string window_name_thermal)
{
  int frame_id = t + INIT_FRAME;

  //char image_name[100];
 // sprintf_s(image_name, 100, "%05d.jpg", frame_id);
  //image = cv::imread(SEQUENCE_PATH + image_name, cv::IMREAD_COLOR);  
  image = cv::imread(frame_files[frame_id-1], IMAGE_TYPE);
  cv::rectangle(image, 
                cv::Rect((int)result_box[t].x, (int)result_box[t].y,  (int)result_box[t].w, (int)result_box[t].h), 
                CV_RGB(255, 255, 0),
                2);

 // sprintf_s(image_name, 100, "#%d", frame_id);
 // cv::putText(image, image_name, cvPoint(0, 60), 2, 2, CV_RGB(255, 255, 0), 3, 0);

  cv::imshow(window_name, image);
  //char buf[100];
  //sprintf_s(buf, "d:\\imgs\\%04d.jpg", frame_id);
  //cv::imwrite(buf, image);
  

  image_thermal = cv::imread(frame_files_thermal[frame_id-1], IMAGE_TYPE_THERMAL);
  cv::rectangle(image_thermal,
	  cv::Rect((int)result_box[t].x, (int)result_box[t].y, (int)result_box[t].w, (int)result_box[t].h),
	  CV_RGB(255, 255, 0),
	  2);
  //cv::imshow(window_name_thermal, image_thermal);

  cv::waitKey(1);
}

void Tracker::run()
{
  std::cout << "[Sequence] " << SEQUENCE_NAME << '\n';
  std::cout << "-------------------------------------------- \n";
  
  /// Initialize 
  std::cout << "Start initialization\n";
  initialize();
  display_result(0, sequence_name1, sequence_name1);
  //cv::waitKey();
  std::cout << "Complete initialization \n";
  std::cout << "-------------------------------------------- \n";

  /// track
  std::cout << "Start tracking\n";
  //initialize the scale model
  initial_scale_model();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int t=1; t<NUM_FRAME; ++t)  
  {   
	  
    int frame_id = t + INIT_FRAME;
	//std::cout << frame_id << std::endl;
	//if (frame_id == 43)
	//{
	//	std::cout << frame_id << std::endl;
	//}
    track(frame_id);
	display_result(t, sequence_name1, sequence_name1);
	frame_idx = frame_id;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto fps = (double)(NUM_FRAME-1.0) / std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count();
  std::cout << "FPS: " << fps << "\n";
  
  std::cout << "Complete tracking \n";
  std::cout << "-------------------------------------------- \n\n";
}

void Tracker::save(std::string cur_path)
{
	//std::ofstream ofs(cur_path + "\\" + RESULT_PATH + "\\bbox\\" + SEQUENCE_NAME + "_ours.txt");// RGB-T
	std::ofstream ofs(RESULT_PATH + "\\bbox\\" + "Ours_"+SEQUENCE_NAME + ".txt");

  for (int i=0; i<result_box.size(); ++i)
  {
    char temp[100];
    sprintf_s(temp, 100, "%.2lf %.2lf %.2lf %.2f %.2lf %.2lf %.2lf %.2f", 
		result_box[i].x + 1, result_box[i].y + 1,
		result_box[i].x + 1 + result_box[i].w, result_box[i].y + 1,
		result_box[i].x + 1 + result_box[i].w, result_box[i].y + 1+result_box[i].h,
		result_box[i].x + 1 , result_box[i].y + 1 + result_box[i].h);   // RGB-T
	//sprintf_s(temp, 100, "%.2lf,%.2lf,%.2lf,%.2f",
	//	result_box[i].x + 1, result_box[i].y + 1, result_box[i].x + result_box[i].w + 1, result_box[i].y + result_box[i].h + 1);  //RGB-D
    ofs << temp << '\n';
  }
  ofs.close(); 
  std::cout << "finish write result_box!" << std::endl;

  //std::ofstream ofs1(RESULT_PATH + "\\query\\" + SEQUENCE_NAME + "_query.txt");
  //for (int i = 0; i < NUM_FRAME; i++)
  //{
	 // ofs1 << query_each_seq[i].transpose() << "\n";
  //}
  //ofs1.close();
  //std::cout << "finish write query!" << std::endl;

  //std::ofstream ofs2(RESULT_PATH + "\\Q\\" + SEQUENCE_NAME + "_q.txt");
  //for (int i = 0; i < NUM_FRAME; i++)
  //{
	 // ofs2 << Q_each_seq[i].transpose() << "\n";
  //}
  //ofs2.close();
  //std::cout << "finish write Q!" << std::endl;

  //std::ofstream ofs3(RESULT_PATH + "\\weight\\" + SEQUENCE_NAME + "_weight.txt");
  ////std::cout << "weight_each_seq: " << weight_each_seq.size() << std::endl;
  //for (int i = 0; i < NUM_FRAME; i++)
  //{
	 // ofs3 << weight_each_seq[i].transpose() << " ";
	 // ofs3 << "\n";
  //}
  //ofs3.close();
  //std::cout << "finish write weight!" << std::endl;

  for (int i = 0; i < MAXITER; i++)
  {
	  iter_errs[i] /= result_box.size();
  }
  std::ofstream ofs4(RESULT_PATH + "\\iter_errs\\" + SEQUENCE_NAME + "_iter_errs.txt");
  for (int i = 0; i < MAXITER; i++)
  {
	  ofs4 << iter_errs[i] << '\n';
  }
  ofs4.close();
  std::cout << "finish write iter_errs!" << std::endl;

  ////std::ofstream ofs5(cur_path + "\\" + RESULT_PATH + "\\W\\" + SEQUENCE_NAME + "_W_v.txt");
  ////std::ofstream ofs5_1(cur_path + "\\" + RESULT_PATH + "\\W\\" + SEQUENCE_NAME + "_W_thermal.txt");
  ////for (int i = 0; i < NUM_FRAME; i++)
  ////{
	 //// ofs5 << W_each_seq[i] << '\n';
	 //// ofs5_1 << W_each_seq_thermal[i] << '\n';
  ////}
  ////ofs5.close();
  ////ofs5_1.close();
  ////std::cout << "finish write W!" << std::endl;

  //std::ofstream ofs6(RESULT_PATH + "\\S\\" + SEQUENCE_NAME + "_S.txt");
  //for (int i = 0; i < NUM_FRAME; i++)
  //{
	 // ofs6 << S_each_seq[i] << '\n';
  //}
  //ofs6.close();
  //std::cout << "finish write S!" << std::endl;

  //std::ofstream ofs7(RESULT_PATH + "\\R\\" + SEQUENCE_NAME + "_R.txt");
  //for (int i = 0; i < NUM_FRAME; i++)
  //{
	 // ofs7 << R_each_seq[i] << '\n';
  //}
  //ofs7.close();
  //std::cout << "finish write R!" << std::endl << std::endl << std::endl;

  query_each_seq.clear();
  Q_each_seq.clear();
  W_each_seq.clear();
  W_each_seq_thermal.clear();
  S_each_seq.clear();
  R_each_seq.clear();
  //patch_weight.clear();
  
}

void Tracker::initialize()
{
	iter_errs.resize(MAXITER);
	mu_errs.resize(MAXITER);

  std::cout << "- initialize feature map\n";
  update_feature_map(INIT_FRAME);

  std::cout << "- initialize patch weight\n";
  initialize_mask();
  initialize_seed();

  update_patch_weight();
   
  std::cout << "- initialize classifier\n";
  update_classifier(true);

  update_result_box(INIT_FRAME);
}

void Tracker::initial_scale_model()
{// caculate the global variable scaleFactors、min_scale_factor、max_scale_factor
	int nScales = numScales;
	scaleFactors.resize(nScales, 0.0);
	double scale_step = Scale_step;

	for (int numi = 1; numi <= nScales; numi++)
	{
		scaleFactors[numi - 1] = pow(scale_step, ceil(nScales / 2.0) - numi);
		//cout << "scale_model_temp = " << scale_model_temp << endl;
	}

	if (feature_bbox.w > feature_bbox.h) {
		min_scale_factor = pow(scale_step, ceil(log(5 / feature_bbox.h) / log(scale_step)));
	}
	else {
		min_scale_factor = pow(scale_step, ceil(log(5 / feature_bbox.w) / log(scale_step)));
	}
	if (image_bbox.w / object_bbox.w > image_bbox.h / object_bbox.h) {
		max_scale_factor = pow(scale_step, floor(log(image_bbox.h / object_bbox.h) / log(scale_step)));
	}
	else {
		max_scale_factor = pow(scale_step, floor(log(image_bbox.w / object_bbox.w) / log(scale_step)));
	}
	min_scale_factor *= 6;

	double PI = 3.1415926, tmp = 0.0;
	scale_window.resize(nScales, 0.0);
	scale_window[0] = 0.00001;//hann()窗正常求值应为0，但0作用特征后输入到分类器做运算结果异常
	for (int N = 1; N < nScales; N++)
	{
		tmp = 0.5 - 0.5*cos(2 * PI * N / (nScales - 1));
		if (N == nScales / 2)
			tmp = 1.0;
		scale_window[N] = tmp;

		//scale_window[N] = (scale_window[N] + 2) / 3;//可提高对尺度敏感性

		//cout << N << ':' << scale_window[N] << endl;
	}
}

void Tracker::update_scale()
{
	//  等比例缩放scale_w,scale_h,search_r；但是patch_w、patch_h不变
	if (best_bbox.w < best_bbox.h)
	{
		scale_w = best_bbox.w / 32.0;
		patch_w = std::round(best_bbox.w / (8.0*scale_w));
		patch_h = std::round(best_bbox.h / (8.0*scale_w));
		scale_h = best_bbox.h / (8.0*patch_h);
	}
	else
	{
		scale_h = best_bbox.h / 32.0;
		patch_h = std::round(best_bbox.h / (8.0*scale_h));
		patch_w = std::round(best_bbox.w / (8.0*scale_h));
		scale_w = best_bbox.w / (8.0*patch_w);
	}
	search_r = sqrt(best_bbox.w*best_bbox.h / (scale_w*scale_h));//搜索区域

}

void Tracker::track(int frame_id)
{
  //clock_t t0 = clock();
  update_feature_map(frame_id);
  //clock_t t1 = clock();
  //std::cout << "feature: " << t1 - t0 << std::endl;
  bool is_updated = update_object_box(frame_id);  
  //std::cout << is_updated << std::endl;

  update_result_box(frame_id); //在此之后修改新的scale_w、 scale_h 和 search_r.

  if (is_updated)
  {  
	  if (is_scale)
	  {
		  update_scale(); //实时判断并更新原始图片的缩放尺度 scale_w/scale_h/search_r

		  //update object_bbox  resize(image) image_bbox feature_bbox  border_bbox

		  //char image_name[100];
		  //sprintf_s(image_name, 100, "%04d.jpg", frame_id);
		  //image = cv::imread(SEQUENCE_PATH + image_name, IMAGE_TYPE);

		  image = cv::imread(frame_files[frame_id - 1], IMAGE_TYPE);
		  cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		  cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());
		  border_bbox.set(0, 0, image.cols - 1, image.rows - 1);
		  image_bbox.set(patch_w, patch_h, border_bbox.w - 2 * patch_w, border_bbox.h - 2 * patch_h);

		  object_bbox.x = std::round(best_bbox.x / scale_w) + patch_w;
		  object_bbox.y = std::round(best_bbox.y / scale_h) + patch_h;
		  object_bbox.w = std::round(best_bbox.w / scale_w);
		  object_bbox.h = std::round(best_bbox.h / scale_h);
		  if (object_bbox.x < 0)
			  object_bbox.x = 0;
		  if (object_bbox.x + object_bbox.w > image_bbox.x + image_bbox.w)
			  object_bbox.x = image_bbox.x + image_bbox.w - object_bbox.w;
		  if (object_bbox.y < 0)
			  object_bbox.y = 0;
		  if (object_bbox.y + object_bbox.h > image_bbox.y + image_bbox.h)
			  object_bbox.y = image_bbox.y + image_bbox.h - object_bbox.h;


		  //2016-09-28
		  //Update patch_mask[64] & expand_patch_mask[100]
		  initialize_mask();

		  //由于跟踪目标的尺度发生变化，分类器更新之前，必须先更新scale/object_bbox等参数
		  update_feature_map(frame_id);
	  }


	//clock_t t0 = clock();
	update_patch_weight();
	//clock_t t1 = clock();
	//std::cout << "weight: " << t1 - t0 << std::endl;
	//t0 = clock();
    update_classifier();   
	//t1 = clock();
	//std::cout << "update: " << t1 - t0 << std::endl;
  }
  /*update_result_box(frame_id);*/
}

void Tracker::update_result_box(int frame_id)
{
  int t = frame_id - INIT_FRAME;
  if (is_scale)
  {
	  result_box[t].x = best_bbox.x;
	  result_box[t].y = best_bbox.y;
	  result_box[t].w = best_bbox.w;
	  result_box[t].h = best_bbox.h;
	  //由于跟踪目标尺度发生变化，所以进入下一帧前 object_bbox 等也要更新，但前提是先计算新的scale_w,scale_h,search_r等尺度参数
  }
  else
  {
	  result_box[t].x = (object_bbox.x - patch_w)*scale_w;
	  result_box[t].y = (object_bbox.y - patch_h)*scale_h;
	  result_box[t].w = object_bbox.w*scale_w;
	  result_box[t].h = object_bbox.h*scale_h;
  }
 // std::cout << "is_Scale : " << is_scale << " best_bbox: " << best_bbox.x << " " << best_bbox.y << " " << best_bbox.w << " " << best_bbox.h << std::endl;
 // std::cout << "is_Scale : " << is_scale << " result_bbox: " << result_box[t].x + 1 << " " << result_box[t].y + 1 << " " << result_box[t].w << " " << result_box[t].h << std::endl;
}

void Tracker::initialize_mask()
{
  patch_mask.clear();
  std::vector<Rect> patch = extract_patch(object_bbox); 
  for (int i=0; i<patch.size(); ++i)
    patch_mask.push_back(Rect(patch[i].x-object_bbox.x, 
                              patch[i].y-object_bbox.y, 
                              patch[i].w, 
                              patch[i].h));
    
  expanded_patch_mask.clear();
  std::vector<Rect> expanded_patch = extract_expanded_patch(object_bbox);
  for (int i=0; i<expanded_patch.size(); ++i)
    expanded_patch_mask.push_back(Rect(expanded_patch[i].x-object_bbox.x,
                                       expanded_patch[i].y-object_bbox.y,
                                       expanded_patch[i].w,
                                       expanded_patch[i].h));

  patch_mask_thermal.clear();

  std::vector<Rect> patch_thermal = extract_patch(object_bbox);

  for (int i = 0; i<patch_thermal.size(); ++i)
	  patch_mask_thermal.push_back(Rect(patch_thermal[i].x - object_bbox.x,
	  patch_thermal[i].y - object_bbox.y,
	  patch_thermal[i].w,
	  patch_thermal[i].h));

  expanded_patch_mask_thermal.clear();

  std::vector<Rect> expanded_patch_thermal = extract_expanded_patch(object_bbox);

  for (int i = 0; i < expanded_patch_thermal.size(); ++i)
	  expanded_patch_mask_thermal.push_back(Rect(expanded_patch_thermal[i].x - object_bbox.x,
	  expanded_patch_thermal[i].y - object_bbox.y,
	  expanded_patch_thermal[i].w,
	  expanded_patch_thermal[i].h));

  //bbox_mask.clear();
  //bbox_mask.push_back(Rect(object_bbox.x, object_bbox.y, bbox_w, bbox_h));		//object_bbox.w  object_bbox.h
  bbox_mask = Rect(object_bbox.x, object_bbox.y, object_bbox.w, object_bbox.h);		//object_bbox.w  object_bbox.h bbox_w, bbox_h
}

void Tracker::initialize_seed()
{  
	std::vector<Rect> expanded_patch = extract_expanded_patch(object_bbox, expanded_patch_mask);
  patch_weight_v.resize(NUM_PATCH, 0.0); 
  patch_weight_i.resize(NUM_PATCH, 0.0);
}

Eigen::MatrixXd Tracker::getMarkMatrix(Eigen::MatrixXd scoreMap, int h, int w)
{
	// get the peaks in the scoreMap
	double maxTmp = 0;
	bool flag_tmp;
	Eigen::MatrixXd markMatrix = Eigen::MatrixXd::Zero(h, w);
	for (int i = 0; i < scoreMap.cols(); i++)
	{
		for (int j = 0; j < scoreMap.rows(); j++)
		{
			if (i == 0 && j == 0) //左上角
			{
				double neighPoint1 = scoreMap(i, j + 1); //右
				double neighPoint2 = scoreMap(i + 1, j); //下
				double neighPoint3 = scoreMap(i + 1, j + 1); // 右下
				flag_tmp = scoreMap(i, j) > neighPoint1 && scoreMap(i, j) > neighPoint2 && scoreMap(i, j) > neighPoint3;
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (i == 0 && j == w - 1) //右上角
			{
				double neighPoint1 = scoreMap(i, j - 1); //左
				double neighPoint2 = scoreMap(i + 1, j - 1); // 左下
				double neighPoint3 = scoreMap(i + 1, j); //下
				flag_tmp = scoreMap(i, j) > neighPoint1 && scoreMap(i, j) > neighPoint2 && scoreMap(i, j) > neighPoint3;
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (i == h - 1 && j == 0) //左下角
			{
				double neighPoint1 = scoreMap(i - 1, j); //上
				double neighPoint2 = scoreMap(i, j + 1); // 右
				double neighPoint3 = scoreMap(i - 1, j + 1); //右上
				flag_tmp = scoreMap(i, j) > neighPoint1 && scoreMap(i, j) > neighPoint2 && scoreMap(i, j) > neighPoint3;
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (i == h - 1 && j == w - 1) //右下角
			{
				double neighPoint1 = scoreMap(i - 1, j); //上
				double neighPoint2 = scoreMap(i, j - 1); //左
				double neighPoint3 = scoreMap(i - 1, j - 1); //左上
				flag_tmp = scoreMap(i, j) > neighPoint1 && scoreMap(i, j) > neighPoint2 && scoreMap(i, j) > neighPoint3;
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (i == 0 && j != 0 && j != w - 1) //上边框
			{
				double neighPoint1 = scoreMap(i, j - 1);//左
				double neighPoint2 = scoreMap(i + 1, j - 1); //左下
				double neighPoint3 = scoreMap(i + 1, j); //下
				double neighPoint4 = scoreMap(i + 1, j + 1); //右下
				double neighPoint5 = scoreMap(i, j + 1);//右
				flag_tmp = (scoreMap(i, j) > neighPoint1) && (scoreMap(i, j) > neighPoint2) && (scoreMap(i, j) > neighPoint3) && (scoreMap(i, j) > neighPoint4) && (scoreMap(i, j) > neighPoint5);
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (i == h - 1 && j != 0 && j != w - 1) //下边框
			{
				double neighPoint1 = scoreMap(i, j - 1);//左
				double neighPoint2 = scoreMap(i - 1, j - 1); //左上
				double neighPoint3 = scoreMap(i - 1, j); //上
				double neighPoint4 = scoreMap(i - 1, j + 1); //右上
				double neighPoint5 = scoreMap(i, j + 1);//右
				flag_tmp = (scoreMap(i, j) > neighPoint1) && (scoreMap(i, j) > neighPoint2) && (scoreMap(i, j) > neighPoint3) && (scoreMap(i, j) > neighPoint4) && (scoreMap(i, j) > neighPoint5);
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (j == 0 && i != 0 && i != h - 1)//左边框
			{
				double neighPoint1 = scoreMap(i - 1, j);//上
				double neighPoint2 = scoreMap(i - 1, j + 1); //右上
				double neighPoint3 = scoreMap(i, j + 1); //右
				double neighPoint4 = scoreMap(i + 1, j + 1); //右下
				double neighPoint5 = scoreMap(i + 1, j);//下
				flag_tmp = (scoreMap(i, j) > neighPoint1) && (scoreMap(i, j) > neighPoint2) && (scoreMap(i, j) > neighPoint3) && (scoreMap(i, j) > neighPoint4) && (scoreMap(i, j) > neighPoint5);
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else if (j == w - 1 && i != 0 && i != h - 1) //右边框
			{
				double neighPoint1 = scoreMap(i - 1, j);//上
				double neighPoint2 = scoreMap(i - 1, j - 1); //左上
				double neighPoint3 = scoreMap(i, j - 1); //左
				double neighPoint4 = scoreMap(i + 1, j - 1); //左下
				double neighPoint5 = scoreMap(i + 1, j);//下
				flag_tmp = (scoreMap(i, j) > neighPoint1) && (scoreMap(i, j) > neighPoint2) && (scoreMap(i, j) > neighPoint3) && (scoreMap(i, j) > neighPoint4) && (scoreMap(i, j) > neighPoint5);
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
			else //中间
			{
				double neighPoint1 = scoreMap(i, j - 1);//左
				double neighPoint2 = scoreMap(i - 1, j - 1); //左上
				double neighPoint3 = scoreMap(i - 1, j); //上
				double neighPoint4 = scoreMap(i - 1, j + 1); //右上
				double neighPoint5 = scoreMap(i, j + 1);//右
				double neighPoint6 = scoreMap(i + 1, j + 1); //右下
				double neighPoint7 = scoreMap(i + 1, j); //下
				double neighPoint8 = scoreMap(i + 1, j - 1);//左下
				flag_tmp = (scoreMap(i, j) > neighPoint1) && (scoreMap(i, j) > neighPoint2) && (scoreMap(i, j) > neighPoint3) && (scoreMap(i, j) > neighPoint4) && (scoreMap(i, j) > neighPoint5) && (scoreMap(i, j) > neighPoint6) && (scoreMap(i, j) > neighPoint7) && (scoreMap(i, j) > neighPoint8);
				markMatrix(i, j) = flag_tmp ? 1 : 0;
			}
		}
	}
	return markMatrix;
}

bool Tracker::isOverBoundry(int index_x, int index_y, int x_shift, int y_shift)
{
	bool flag = false; //默认未超出边界
	int x = index_x - search_r + x_shift;
	int y = index_y - search_r + y_shift;
	if (x < -32 || x > 32 || y < -32 || y > 32)
	{
		flag = true;
	}
	return flag;
}

bool Tracker::isTruePeak(Eigen::MatrixXd scoreMap, double peak, int index_x, int index_y)
{
	int winSize = 5;
	int xShift[5] = { -2, -1, 0, 1, 2 };
	int yShift[5] = { -2, -1, 0, 1, 2 };
	//int x_new, y_new;
	//double maxPeak = 0;
	double neighPoint;
	bool flag = false; // 判断是否越界
	bool isPeak = false; // 判断是否是真正的peak

	for (int i = 0; i < winSize; i++)
	{
		for (int j = 0; j < winSize; j++)
		{
			// 判断flag
			flag = isOverBoundry(index_x, index_y, xShift[i], yShift[j]);
			if (!flag)
			{
				neighPoint = scoreMap(index_x + xShift[i], index_y + yShift[j]);
				// 只要有一个邻域值比该值大，就移除此极值点
				if (peak <= neighPoint)
					isPeak = false;
				else
					isPeak = true;
			}

		}
	}
	return isPeak;
}


bool Tracker::update_object_box(int frame_id)
{
  bool is_updated = false;

  Rect sample(object_bbox);
  Rect best_sample(object_bbox);
  double best_score = -DBL_MAX;	  

  //多峰检测
  /*double maxPeak = 0;
  int maxPeak_x;
  int maxPeak_y;
  double secPeak = 0;
  Eigen::MatrixXd scoreMap = Eigen::MatrixXd::Zero(2 * search_r + 1, 2 * search_r + 1);
  Eigen::MatrixXd markB = Eigen::MatrixXd::Zero(2 * search_r + 1, 2 * search_r + 1);
  Eigen::MatrixXd markB_new = Eigen::MatrixXd::Zero(2 * search_r + 1, 2 * search_r + 1);

  Eigen::MatrixXd optScoreMap_first = Eigen::MatrixXd::Zero(2 * search_r + 1, 2 * search_r + 1);
  Eigen::MatrixXd optScoreMap_second = Eigen::MatrixXd::Zero(2 * search_r + 1, 2 * search_r + 1);

  bool isPeak;*/


  for (int iy = -search_r; iy <= search_r; ++iy)
  {
	  for (int ix = -search_r; ix <= search_r; ++ix)
	  {
		  sample.x = (int)object_bbox.x + ix;
		  sample.y = (int)object_bbox.y + iy;     

		  if (!sample.is_inside(image_bbox))  
			  continue; 

		  Eigen::VectorXd sample_feature = extract_test_feature(sample); 
		  double score = (1 - OMEGA)*classifier.test(sample_feature) + OMEGA*classifier0.test(sample_feature);
		  if (score > best_score)
		  {
			  best_score = score;
			  best_sample.set(sample);
		  }
		  //scoreMap(iy + search_r, ix + search_r) = score;
	  }
  }
  
  //markB = getMarkMatrix(scoreMap, 2 * search_r + 1, 2 * search_r + 1);
  ////std::ofstream ofs_score("D:\\scoreMap.txt");
  ////ofs_score << scoreMap;
  ////ofs_score.close();

  ////int temp = 0;
  //// 找所有局部极值点 并确定其是否为局部值
  //for (int i = 0; i < scoreMap.rows(); i++)
  //{
	 // for (int j = 0; j < scoreMap.cols(); j++)
	 // {
		//  optScoreMap_first(i, j) = scoreMap(i, j) * markB(i, j);
		//  optScoreMap_second(i, j) = optScoreMap_first(i, j);
		//  if (optScoreMap_first(i, j) >= maxPeak)
		//  {
		//	  maxPeak = optScoreMap_first(i, j);
		//	  maxPeak_x = j;//i;
		//	  maxPeak_y = i;//j;
		//  }
		//  if (optScoreMap_first(i, j) != 0)
		//  {
		//	  //temp++;
		//	  isPeak = isTruePeak(scoreMap, optScoreMap_first(i, j), i, j);
		//	  //std::cout << temp << " Peak: " << isPeak << "\n";
		//	  if (!isPeak)
		//	  {
		//		  optScoreMap_second(i, j) = 0;
		//	  }
		//  }
	 // }
  //}

  //// 找第二大值
  //for (int i = 0; i < scoreMap.rows(); i++)
  //{
	 // for (int j = 0; j < scoreMap.cols(); j++)
	 // {
		//  if (optScoreMap_second(i, j) != maxPeak)
		//  {
		//	  secPeak = secPeak > optScoreMap_second(i, j) ? secPeak : optScoreMap_second(i, j);
		//  }
	 // }
  //}
  //// 判断是否为全局峰值
  ////if ((maxPeak - secPeak) > PeakThreshold)
  ////std::cout << "maxPeak: " << maxPeak << " secPeak: " << secPeak << " ratio: " << secPeak / maxPeak << "\n";
  //


  Eigen::VectorXd best_sample_feature = extract_test_feature(best_sample); 
  double validation_score = classifier.validation_test(best_sample_feature); 
  //std::cout << "validation_score: " << validation_score << std::endl;
  if (validation_score > THETA)
	  //if (validation_score > THETA && !((secPeak * 1.0 / maxPeak) > PeakThreshold))
  {    
	  object_temp_bbox.x = (best_sample.x - patch_w)*scale_w;
	  object_temp_bbox.y = (best_sample.y - patch_h)*scale_h;
	  object_temp_bbox.w = best_sample.w*scale_w;
	  object_temp_bbox.h = best_sample.h*scale_h;

	  //std::cout << "best:" << best_bbox.x << " " << best_bbox.y << " " << best_bbox.w << " " << best_bbox.h << " " << std::endl;

	  //先恢复到原图大小，在原图分辨率上做尺度变化
	  if (frame_id % interval == 0)
	      scale_estimation(frame_id, object_temp_bbox);

	  double max_scale_score = 0;
	  int num = 0, max_scale_score_num = 0;
	  double w_temp = 0, h_temp = 0;
	  double min_w_temp = 0, min_h_temp = 0;
	  double rate_temp;
	  if (validation_score_s.size())
	  {
		  max_scale_score = *validation_score_s.begin();
		  std::vector<double>::iterator iter;
		  for (iter = validation_score_s.begin(); iter != validation_score_s.end(); iter++)
		  {
			  if (max_scale_score < *iter && num != (numScales - 1) / 2)
			  {
				  max_scale_score = *iter;
				  max_scale_score_num = num;
			  }
			  num++;
		  }
	  }

	  //std::cout << frame_id << '/' << NUM_FRAME << ": " << validation_score << ',' << max_scale_score << std::endl;
	  if (!validation_score_s.empty())
	  {
		  validation_score = validation_score_s[int((numScales - 1) / 2)];//由于特征金字塔里倍数为1.0时的分类器得分（因为新的方式里特征提取后做归一化时与原先归一化结果不同）
	  }
	  //if (frame_id % interval == 0)
		 // std::cout << frame_id << '/' << NUM_FRAME << ": " << validation_score << ',' << max_scale_score << std::endl;

	  if (max_scale_score - validation_score > IsScale_Threshold)
	  {
		 // std::cout << "isScale: True" << std::endl;
		  is_scale = true;

		  //best_bbox = best_sample_s.at(max_scale_score_num);
		  //访问容器里的元素，若使用at()，需执行范围检查，如果参数无效，at()就会抛出一个std::out_of_range异常
		  best_bbox = best_sample_s[max_scale_score_num];

		  //std::cout << "best:" << best_bbox.x << " " << best_bbox.y << " " << best_bbox.w << " " << best_bbox.h << " " << std::endl;

		  currentScaleFactor *= scaleFactors[max_scale_score_num];
		  if (currentScaleFactor < min_scale_factor)
		  {
			  currentScaleFactor = min_scale_factor;
			  is_scale = false;
			  object_bbox.set(best_sample); // non-scale variety
		  }
		  else
			  if (currentScaleFactor > max_scale_factor)
			  {
				  currentScaleFactor = max_scale_factor;
				  is_scale = false;
				  object_bbox.set(best_sample); // non-scale variety
			  }
	  }
	  //cout << best_bbox.x << ' ' << best_bbox.y << ' ' << best_bbox.w << ' ' << best_bbox.h << endl;	
	  //然后，比较 best_sample 和 best_sample_s 分类器得分
	  else
	  {
		  is_scale = false;
		  object_bbox.set(best_sample); // non-scale variety

		  /*is_updated = true;*/
	  }
	  //进入下一帧之前，清空 validation_score_s 和 best_sample_s 容器里的内容
	  validation_score_s.clear();
	  best_sample_s.clear();

	  is_updated = true;
  }

  //std::cout << is_updated << std::endl;
  return is_updated;
}

Eigen::VectorXd Tracker::extract_test_feature_s(cv::Mat &expand_roi, cv::Mat &expand_roi_thermal)
{
	//Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * OBJECT_DIM));//新加入thermal特征，特征维度*2
	Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * OBJECT_DIM + 2 * BBOX_DIM));//新加入thermal特征，特征维度*2
	for (int i = 0; i<NUM_PATCH; ++i)
	{
		int x_min = 0 + patch_mask[i].x;
		int y_min = 0 + patch_mask[i].y;

		feature.segment(i*PATCH_DIM * 2, PATCH_DIM) = patch_weight_v[i] * feature_map_s[expand_roi.cols*y_min + x_min];
		feature.segment(i*PATCH_DIM * 2 + PATCH_DIM, PATCH_DIM) = patch_weight_i[i] * feature_map_thermal_s[expand_roi_thermal.cols*y_min + x_min];//新加入thermal特征
	}

	Rect r_bbox(object_bbox.x, object_bbox.y, object_bbox.w, object_bbox.h);
	feature.segment(OBJECT_DIM * 2, BBOX_DIM) = global_ration*feature_map_bbox_s[expand_roi.cols*(expand_roi.rows-1) + expand_roi.cols - 1];
	feature.segment(OBJECT_DIM * 2 + BBOX_DIM, BBOX_DIM) = global_ration*feature_map_bbox_thermal_s[expand_roi_thermal.cols * (expand_roi_thermal.rows - 1) + expand_roi_thermal.cols - 1];

	//feature.segment(OBJECT_DIM * 2, BBOX_DIM) = global_ration*feature_map_bbox_s[expand_roi.cols*bbox_mask.y + bbox_mask.x];
	//feature.segment(OBJECT_DIM * 2 + BBOX_DIM, BBOX_DIM) = global_ration*feature_map_bbox_thermal_s[expand_roi_thermal.cols * bbox_mask.y + bbox_mask.x];


	feature.normalize();// 向量归一化？？？ 与在搜索窗口里固定尺度对当前位置提取的特征（可能）不一样！！！
	return feature;
}

void Tracker::compute_color_histogram_map_s(cv::Mat &expand_roi1, cv::Mat &expand_roi_thermal)
{
	double bin_size = 32.0;
	int num_color_channel = NUM_CHANNEL - 1;

	for (int i = 0; i<num_color_channel; ++i)
	{
		cv::Mat tmp(expand_roi1.rows, expand_roi1.cols, CV_8UC1);
		tmp.setTo(0);

		for (int j = 0; j<CHANNEL_DIM; ++j)
		{
			for (int y = 0; y<expand_roi1.rows; ++y)
			{
				const uchar* src = image_channel_s[i].ptr(y);
				uchar* dst = tmp.ptr(y);
				for (int x = 0; x<expand_roi1.cols; ++x)
				{
					int bin = (int)((double)(*src) / bin_size);
					*dst = (bin == j) ? 1 : 0;
					++src;
					++dst;
				}
			}

			cv::integral(tmp, integ_hist_s[i*CHANNEL_DIM + j]);
		}
		/************** 新加入的热红外thermal **********/
		cv::Mat tmp_thermal(expand_roi_thermal.rows, expand_roi_thermal.cols, CV_8UC1);
		tmp_thermal.setTo(0);
		for (int j = 0; j<CHANNEL_DIM; ++j)
		{
			for (int y = 0; y<expand_roi_thermal.rows; ++y)
			{
				const uchar* src_thermal = image_channel_thermal_s[i].ptr(y);
				uchar*       dst_thermal = tmp_thermal.ptr(y);

				for (int x = 0; x<expand_roi_thermal.cols; ++x)
				{
					int bin_thermal = (int)((double)(*src_thermal) / bin_size);
					*dst_thermal = (bin_thermal == j) ? 1 : 0;
					++src_thermal;
					++dst_thermal;
				}
			}
			cv::integral(tmp_thermal, integ_hist_thermal_s[i*CHANNEL_DIM + j]);
		}
		/************** 新加入的热红外thermal **********/
	}
}

void Tracker::compute_gradient_histogram_map_s(cv::Mat &expand_roi1, cv::Mat &expand_roi_thermal)
{
	float bin_size = 22.5;
	float radian_to_degree = 180.0 / CV_PI;

	cv::Mat gray_image(expand_roi1.rows, expand_roi1.cols, CV_8UC1);
	if (IMAGE_TYPE == cv::IMREAD_COLOR)
		cv::cvtColor(expand_roi1, gray_image, CV_BGR2GRAY);
	else
		expand_roi1.copyTo(gray_image);

	cv::Mat x_sobel, y_sobel;
	cv::Sobel(gray_image, x_sobel, CV_32FC1, 1, 0);
	cv::Sobel(gray_image, y_sobel, CV_32FC1, 0, 1);

	std::vector<cv::Mat> bins;
	for (int i = 0; i<CHANNEL_DIM; ++i)
		bins.push_back(cv::Mat::zeros(expand_roi1.rows, expand_roi1.cols, CV_32FC1));

	for (int y = 0; y<expand_roi1.rows; ++y)
	{
		float* x_sobel_row_ptr = (float*)(x_sobel.row(y).data);
		float* y_sobel_row_ptr = (float*)(y_sobel.row(y).data);

		std::vector<float*> bins_row_ptrs(CHANNEL_DIM, nullptr);
		for (int i = 0; i<CHANNEL_DIM; ++i)
			bins_row_ptrs[i] = (float*)(bins[i].row(y).data);

		for (int x = 0; x<expand_roi1.cols; ++x)
		{
			if (x_sobel_row_ptr[x] == 0)
				x_sobel_row_ptr[x] += 0.00001;

			float orientation = atan(y_sobel_row_ptr[x] / x_sobel_row_ptr[x])*radian_to_degree + 90;
			float magnitude = sqrt(x_sobel_row_ptr[x] * x_sobel_row_ptr[x] + y_sobel_row_ptr[x] * y_sobel_row_ptr[x]);

			for (int i = 1; i<CHANNEL_DIM; ++i)
			{
				if (orientation <= bin_size*i)
				{
					bins_row_ptrs[i - 1][x] = magnitude;
					break;
				}
			}
		}
	}

	/************** 新加入的thermal **************/
	cv::Mat gray_image_thermal(expand_roi_thermal.rows, expand_roi_thermal.cols, CV_8UC1);
	if (IMAGE_TYPE == cv::IMREAD_COLOR)
		cv::cvtColor(expand_roi_thermal, gray_image_thermal, CV_BGR2GRAY);
	else
		expand_roi_thermal.copyTo(gray_image_thermal);

	cv::Mat x_sobel_thermal, y_sobel_thermal;
	cv::Sobel(gray_image_thermal, x_sobel_thermal, CV_32FC1, 1, 0);
	cv::Sobel(gray_image_thermal, y_sobel_thermal, CV_32FC1, 0, 1);

	std::vector<cv::Mat> bins_thermal;
	for (int i = 0; i<CHANNEL_DIM; ++i)
		bins_thermal.push_back(cv::Mat::zeros(expand_roi_thermal.rows, expand_roi_thermal.cols, CV_32FC1));

	for (int y = 0; y<expand_roi_thermal.rows; ++y)
	{
		float* x_sobel_row_ptr_thermal = (float*)(x_sobel_thermal.row(y).data);
		float* y_sobel_row_ptr_thermal = (float*)(y_sobel_thermal.row(y).data);

		std::vector<float*> bins_row_ptrs_thermal(CHANNEL_DIM, nullptr);
		for (int i = 0; i<CHANNEL_DIM; ++i)
			bins_row_ptrs_thermal[i] = (float*)(bins_thermal[i].row(y).data);

		for (int x = 0; x<expand_roi_thermal.cols; ++x)
		{
			if (x_sobel_row_ptr_thermal[x] == 0)
				x_sobel_row_ptr_thermal[x] += 0.00001;

			float orientation = atan(y_sobel_row_ptr_thermal[x] / x_sobel_row_ptr_thermal[x])*radian_to_degree + 90;
			float magnitude = sqrt(x_sobel_row_ptr_thermal[x] * x_sobel_row_ptr_thermal[x] + y_sobel_row_ptr_thermal[x] * y_sobel_row_ptr_thermal[x]);

			for (int i = 1; i<CHANNEL_DIM; ++i)
			{
				if (orientation <= bin_size*i)
				{
					bins_row_ptrs_thermal[i - 1][x] = magnitude;
					break;
				}
			}
		}
	}
	/************** 新加入的thermal **************/

	int color_dim = PATCH_DIM - CHANNEL_DIM;
	for (int i = 0; i<CHANNEL_DIM; ++i)
	{
		cv::integral(bins[i], integ_hist_s[color_dim + i]);
		cv::integral(bins_thermal[i], integ_hist_thermal_s[color_dim + i]);
	}
}

Eigen::VectorXd Tracker::extract_patch_feature_s(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature(PATCH_DIM);

	int color_dim = PATCH_DIM - CHANNEL_DIM;
	double patch_area = patch_w*patch_h;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist_s[i].at<int>(y_min, x_min)
			+ integ_hist_s[i].at<int>(y_max, x_max)
			- integ_hist_s[i].at<int>(y_max, x_min)
			- integ_hist_s[i].at<int>(y_min, x_max);
		//if (frame_idx == 97)
		//	cout << "sum = " << sum << ';' << x_min << ',' << y_min << ',' << x_max << ',' << y_max << endl;
		feature[i] = sum / patch_area;
	}
	double total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist_s[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist_s[color_dim + i].at<double>(y_max, x_max)
			- integ_hist_s[color_dim + i].at<double>(y_max, x_min)
			- integ_hist_s[color_dim + i].at<double>(y_min, x_max);

		feature[color_dim + i] = sum;
		total_sum += sum;
	}
	for (int i = 0; i<grad_dim; ++i)
		feature[color_dim + i] /= (total_sum + 1e-6);

	return feature;
}

Eigen::VectorXd Tracker::extract_patch_feature_thermal_s(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature(PATCH_DIM);

	int color_dim = PATCH_DIM - CHANNEL_DIM;
	double patch_area = patch_w*patch_h;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist_thermal_s[i].at<int>(y_min, x_min)
			+ integ_hist_thermal_s[i].at<int>(y_max, x_max)
			- integ_hist_thermal_s[i].at<int>(y_max, x_min)
			- integ_hist_thermal_s[i].at<int>(y_min, x_max);
		//if (frame_idx == 97)
		//	cout << "sum = " << sum << ';' << x_min << ',' << y_min << ',' << x_max << ',' << y_max << endl;
		feature[i] = sum / patch_area;
	}
	double total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist_thermal_s[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist_thermal_s[color_dim + i].at<double>(y_max, x_max)
			- integ_hist_thermal_s[color_dim + i].at<double>(y_max, x_min)
			- integ_hist_thermal_s[color_dim + i].at<double>(y_min, x_max);

		feature[color_dim + i] = sum;
		total_sum += sum;
	}
	for (int i = 0; i<grad_dim; ++i)
		feature[color_dim + i] /= (total_sum + 1e-6);

	return feature;
}

Eigen::VectorXd Tracker::extract_bbox_feature_s(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature(BBOX_DIM);

	int color_dim = BBOX_DIM - CHANNEL_DIM;
	//double patch_area = patch_w*patch_h;
	double bbox_area = bbox_w * bbox_h;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist_s[i].at<int>(y_min, x_min)
			+ integ_hist_s[i].at<int>(y_max, x_max)
			- integ_hist_s[i].at<int>(y_max, x_min)
			- integ_hist_s[i].at<int>(y_min, x_max);
		//if (frame_idx == 97)
		//	cout << "sum = " << sum << ';' << x_min << ',' << y_min << ',' << x_max << ',' << y_max << endl;
		feature[i] = sum / bbox_area; // patch_area;
	}
	double total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist_s[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist_s[color_dim + i].at<double>(y_max, x_max)
			- integ_hist_s[color_dim + i].at<double>(y_max, x_min)
			- integ_hist_s[color_dim + i].at<double>(y_min, x_max);

		feature[color_dim + i] = sum;
		total_sum += sum;
	}
	for (int i = 0; i<grad_dim; ++i)
		feature[color_dim + i] /= (total_sum + 1e-6);

	return feature;
}

Eigen::VectorXd Tracker::extract_bbox_feature_thermal_s(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature(BBOX_DIM);

	int color_dim = BBOX_DIM - CHANNEL_DIM;
	//double patch_area = patch_w*patch_h;
	double bbox_area = bbox_w * bbox_h;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist_thermal_s[i].at<int>(y_min, x_min)
			+ integ_hist_thermal_s[i].at<int>(y_max, x_max)
			- integ_hist_thermal_s[i].at<int>(y_max, x_min)
			- integ_hist_thermal_s[i].at<int>(y_min, x_max);
		//if (frame_idx == 97)
		//	cout << "sum = " << sum << ';' << x_min << ',' << y_min << ',' << x_max << ',' << y_max << endl;
		//feature[i] = sum / patch_area;
		feature[i] = sum / bbox_area;
	}
	double total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist_thermal_s[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist_thermal_s[color_dim + i].at<double>(y_max, x_max)
			- integ_hist_thermal_s[color_dim + i].at<double>(y_max, x_min)
			- integ_hist_thermal_s[color_dim + i].at<double>(y_min, x_max);

		feature[color_dim + i] = sum;
		total_sum += sum;
	}
	for (int i = 0; i<grad_dim; ++i)
		feature[color_dim + i] /= (total_sum + 1e-6);

	return feature;
}



void Tracker::compute_feature_map_s(cv::Mat &expand_roi1, cv::Mat &expand_roi_thermal)
{
	for (int iy = 0; iy <= object_bbox.h; iy++)
	{
		int y_min = iy;
		int y_max = y_min + patch_h;
		if (y_max > expand_roi1.rows) { continue; }
		for (int ix = 0; ix <= object_bbox.w; ix++)
		{
			int x_min = ix;
			int x_max = x_min + patch_w;
			if (x_max > expand_roi1.cols) { continue; }
			//if (frame_idx == 50)
			//{
			//	cout << x_min << ',' << y_min << ',' << x_max << ',' << y_max << ',' << endl;
			//}
			feature_map_s[expand_roi1.cols*y_min + x_min] = extract_patch_feature_s(x_min, y_min, x_max, y_max);
			feature_map_thermal_s[expand_roi1.cols*y_min + x_min] = extract_patch_feature_thermal_s(x_min, y_min, x_max, y_max);
		}
	}
	// 提取全局的
	for (int iy = 0; iy <= object_bbox.h; iy++)
	{
		int y_min = iy;
		int y_max = y_min + bbox_h;
		if (y_max > expand_roi1.rows) { continue; }
		for (int ix = 0; ix <= object_bbox.w; ix++)
		{
			int x_min = ix;
			int x_max = x_min + bbox_w;
			if (x_max > expand_roi1.cols) { continue; }
			//if (frame_idx == 50)
			//{
			//	cout << x_min << ',' << y_min << ',' << x_max << ',' << y_max << ',' << endl;
			//}
			feature_map_bbox_s[expand_roi1.cols*y_min + x_min] = extract_bbox_feature_s(x_min, y_min, x_max, y_max);
			feature_map_bbox_thermal_s[expand_roi1.cols*y_min + x_min] = extract_bbox_feature_thermal_s(x_min, y_min, x_max, y_max);
		}
	}
}

void Tracker::extract_sample_feature_s(cv::Mat &expand_roi1, cv::Mat &expand_roi_thermal)
{
	image_channel_s.clear();
	int num_color_channel = NUM_CHANNEL - 1;
	for (int i = 0; i < num_color_channel; ++i)
		image_channel_s.push_back(cv::Mat(expand_roi1.rows, expand_roi1.cols, CV_8UC1));

	integ_hist_s.clear();
	int color_dim = PATCH_DIM - CHANNEL_DIM;
	for (int i = 0; i < color_dim; ++i)
		integ_hist_s.push_back(cv::Mat(expand_roi1.rows + 1, expand_roi1.cols + 1, CV_32SC1));
	for (int i = 0; i < CHANNEL_DIM; ++i)
		integ_hist_s.push_back(cv::Mat(expand_roi1.rows + 1, expand_roi1.cols + 1, CV_64FC1));

	feature_map_s.resize(expand_roi1.rows*expand_roi1.cols, Eigen::VectorXd::Zero(PATCH_DIM));
	feature_map_bbox_s.resize(expand_roi1.rows*expand_roi1.cols, Eigen::VectorXd::Zero(BBOX_DIM)); // 提取全局
	cv::split(expand_roi1, image_channel_s);

	/**************************** 新加入thermal *******************************/
	image_channel_thermal_s.clear();
	//int num_color_channel = NUM_CHANNEL - 1;
	for (int i = 0; i < num_color_channel; ++i)
		image_channel_thermal_s.push_back(cv::Mat(expand_roi_thermal.rows, expand_roi_thermal.cols, CV_8UC1));

	integ_hist_thermal_s.clear();
	//int color_dim = PATCH_DIM - CHANNEL_DIM;
	for (int i = 0; i < color_dim; ++i)
		integ_hist_thermal_s.push_back(cv::Mat(expand_roi_thermal.rows + 1, expand_roi_thermal.cols + 1, CV_32SC1));
	for (int i = 0; i < CHANNEL_DIM; ++i)
		integ_hist_thermal_s.push_back(cv::Mat(expand_roi_thermal.rows + 1, expand_roi_thermal.cols + 1, CV_64FC1));

	feature_map_thermal_s.resize(expand_roi_thermal.rows*expand_roi_thermal.cols, Eigen::VectorXd::Zero(PATCH_DIM));
	feature_map_bbox_thermal_s.resize(expand_roi_thermal.rows*expand_roi_thermal.cols, Eigen::VectorXd::Zero(BBOX_DIM)); //全局
	cv::split(expand_roi_thermal, image_channel_thermal_s);
	/**************************** 新加入thermal *******************************/

	//if (frame_idx == 16)
	//	cout << "debugging" << endl;
	compute_color_histogram_map_s(expand_roi1, expand_roi_thermal);
	compute_gradient_histogram_map_s(expand_roi1, expand_roi_thermal);
	compute_feature_map_s(expand_roi1, expand_roi_thermal);

}

void Tracker::scale_estimation(int frame_id, const Rect &sample)
{
	double scale_score;
	double x_min, y_min, x_max, y_max;
	double x_min_e, y_min_e, x_max_e, y_max_e;
	double scale_w_s, scale_h_s;
	Rect temp_sample_s;

	char image_name[100];
	cv::Mat image_original = cv::imread(frame_files[frame_id - 1], IMAGE_TYPE);
	cv::Mat image_original_tmp = image_original;

	/********* 新加入thermal项 ***********/
	cv::Mat image_original_thermal = cv::imread(frame_files_thermal[frame_id - 1], IMAGE_TYPE);
	if (SEQUENCE_NAME == "occBike")
	{
		cv::copyMakeBorder(image_original_thermal, image_original_thermal, 0, 8, 0, 16, cv::BORDER_CONSTANT, cv::Scalar());
	}
	cv::Mat image_original_tmp_thermal = image_original_thermal;
	/********* 新加入thermal项 ***********/

	validation_score_s.clear();
	best_sample_s.clear();

		scale_model_id = 0;
		double scale_alph;//假设尺度按比例scale_alph缩放；W = scale_alph * W, H = scale_alph * H;
		std::vector<double>::iterator st;
		int numScale = 0;
		for (st = scaleFactors.begin(); st != scaleFactors.end(); st++)
		{
			scale_model_id++;
			numScale++;
			scale_alph = *st;
			
			x_min = sample.x + 1 - sample.w*(scale_alph - 1) / 2;
			y_min = sample.y + 1 - sample.h*(scale_alph - 1) / 2;//为什么要+1呢？ 不加1的话，Car4差别很明显——20170104
			
			if (x_min < 0 || y_min < 0) //此处，注意与当前分辨率下那种处理方式的细微区别
			{
				if (scale_alph != 1){
					scale_score = 0;
					validation_score_s.push_back(scale_score);
					best_sample_s.push_back(sample);
					continue;
				}
			}         //是否越界 //保证容器size始终等于模板数
			
			x_max = x_min + scale_alph*sample.w;
			y_max = y_min + scale_alph*sample.h;
			
			if (x_max > image_original.cols - 1 || y_max > image_original.rows - 1)
			{
				if (scale_alph != 1){
				scale_score = 0; 
				validation_score_s.push_back(scale_score); 
				best_sample_s.push_back(sample);
				continue;
				}
			}   //是否越界
			cv::Mat roi(image_original, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min));//特别注意，该命令“取整”
			//cv::namedWindow("test...");
			//cv::imshow("test...", roi);
			temp_sample_s.x = x_min;
			temp_sample_s.y = y_min;
			temp_sample_s.w = scale_alph*sample.w;
			temp_sample_s.h = scale_alph*sample.h;

			//提取的尺度金字塔全部resize成当前帧object_bbox默认大小矩形框
			scale_w_s = object_bbox.w / roi.cols;
			scale_h_s = object_bbox.h / roi.rows;
			//cv::resize(roi, roi, cv::Size(), scale_w_s, scale_h_s);

			//if (frame_id == 792)
			//	std::cout << std::endl;

			double patch_w_tmp, patch_h_tmp;
			patch_w_tmp = patch_w;
			patch_h_tmp = patch_h;

			//std::cout << image_original.cols << ',' << image_original.rows << std::endl;
			//std::cout << patch_w_tmp / scale_w_s << ',' << patch_h_tmp / scale_h_s << std::endl;
			cv::copyMakeBorder(image_original, image_original_tmp, patch_h_tmp / scale_h_s, patch_h_tmp / scale_h_s, patch_w_tmp / scale_w_s, patch_w_tmp / scale_w_s, cv::BORDER_CONSTANT, cv::Scalar());
			//std::cout << image_original_tmp.cols << ',' << image_original_tmp.rows << std::endl;

			//extract RGB+HOG 32D features and save it
			/*MakeBorder拓展图片的原因，必须动态更新矩形框坐标信息*/
			x_min_e = int(x_min) + patch_w_tmp / scale_w_s;	//cv::Mat roi已经相当于对x_min取整
			y_min_e = int(y_min) + patch_h_tmp / scale_h_s;	//cv::Mat roi已经相当于对y_min取整
			/*目标矩形框特征描述是通过计算积分图像而获得的原因，必须在 右下 方多提取一个积分单位的信息*/
			x_max_e = x_min_e + int(x_max - x_min) + patch_w_tmp / scale_w_s;	//cv::Mat roi已经相当于对x_max-x_min取整
			y_max_e = y_min_e + int(y_max - y_min) + patch_h_tmp / scale_h_s; //cv::Mat roi已经相当于对y_max-y_min取整

			cv::Mat expand_roi(image_original_tmp, cv::Rect(x_min_e, y_min_e, x_max_e - x_min_e, y_max_e - y_min_e));
			cv::copyMakeBorder(expand_roi, expand_roi, 0, patch_h_tmp / scale_h_s, 0, patch_w_tmp / scale_w_s, cv::BORDER_CONSTANT, cv::Scalar());
			cv::resize(expand_roi, expand_roi, cv::Size(), scale_w_s, scale_h_s);
			//expand_roi:宽和高分别加上1倍的patch_w和patch_h，这是因为extract_patch_feature()里提roi矩形框里的特征需要

			/********* 新加入thermal项 ***********/
			cv::copyMakeBorder(image_original_thermal, image_original_tmp_thermal, patch_h_tmp / scale_h_s, patch_h_tmp / scale_h_s, patch_w_tmp / scale_w_s, patch_w_tmp / scale_w_s, cv::BORDER_CONSTANT, cv::Scalar());
			cv::Mat expand_roi_thermal(image_original_tmp_thermal, cv::Rect(x_min_e, y_min_e, x_max_e - x_min_e, y_max_e - y_min_e));
			cv::copyMakeBorder(expand_roi_thermal, expand_roi_thermal, 0, patch_h_tmp / scale_h_s, 0, patch_w_tmp / scale_w_s, cv::BORDER_CONSTANT, cv::Scalar());
			cv::resize(expand_roi_thermal, expand_roi_thermal, cv::Size(), scale_w_s, scale_h_s);
			/********* 新加入thermal项 ***********/

			extract_sample_feature_s(expand_roi, expand_roi_thermal);
			//Eigen::VectorXd scale_sample_feature = extract_test_feature_s(expand_roi, expand_roi_thermal);
			Eigen::VectorXd scale_sample_feature = extract_test_feature_s(expand_roi, expand_roi_thermal) * scale_window[numScale - 1];

			//scale_score = classifier.validation_test(scale_sample_feature);
			scale_score = (1 - OMEGA)*classifier.validation_test(scale_sample_feature) + OMEGA*classifier0.validation_test(scale_sample_feature);

			//Long-term correlation tracking: 33层尺度金字塔特征还要乘以相应尺度系数scale_window（hann(33)计算所得
			//scale_score *= scale_window[numScale - 1];//给予不同尺度模板发生可能性的概率：尺度渐变的可能性大，尺度突变的可能性相对较低

			validation_score_s.push_back(scale_score);
			best_sample_s.push_back(temp_sample_s);
		}

	//运行至此处，每个scale变化模板求得的分数按顺序依次保存在 <double>Tracker::validation_score_s容器中
	//对应在原图上boundingbox的结果保存在 <Rect> Tracker::best_sample_s容器里。

}

void Tracker::update_feature_map(int idx)
{
	//char image_name[100];
	//sprintf_s(image_name, 100, "%04d.jpg", idx);
	//image = cv::imread(SEQUENCE_PATH + image_name, IMAGE_TYPE);

	image = cv::imread(frame_files[idx - 1], IMAGE_TYPE);
	cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
	cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

	image_channel.clear();
	int num_color_channel = NUM_CHANNEL - 1;
	for (int i = 0; i<num_color_channel; ++i)
		image_channel.push_back(cv::Mat(image.rows, image.cols, CV_8UC1));

	integ_hist.clear();
	int color_dim = PATCH_DIM - CHANNEL_DIM;
	for (int i = 0; i<color_dim; ++i)
		integ_hist.push_back(cv::Mat(image.rows + 1, image.cols + 1, CV_32SC1));
	for (int i = 0; i<CHANNEL_DIM; ++i)
		integ_hist.push_back(cv::Mat(image.rows + 1, image.cols + 1, CV_64FC1));

	feature_map.resize(image.rows*image.cols, Eigen::VectorXd::Zero(PATCH_DIM));
	feature_map_bbox.resize(image.rows*image.cols, Eigen::VectorXd::Zero(BBOX_DIM)); // 提取全局的
	cv::split(image, image_channel);

	/****************** 加入热红外thermal *********************/
	image_thermal = cv::imread(frame_files_thermal[idx - 1], IMAGE_TYPE_THERMAL);
	if (SEQUENCE_NAME == "occBike")
	{
		cv::copyMakeBorder(image_thermal, image_thermal, 0, 8, 0, 16, cv::BORDER_CONSTANT, cv::Scalar());
	}

	cv::resize(image_thermal, image_thermal, cv::Size(), 1 / scale_w, 1 / scale_h);
	cv::copyMakeBorder(image_thermal, image_thermal, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

	image_channel_thermal.clear();
	//int num_color_channel = NUM_CHANNEL - 1;
	for (int i = 0; i<num_color_channel; ++i)
		image_channel_thermal.push_back(cv::Mat(image_thermal.rows, image_thermal.cols, CV_8UC1));

	integ_hist_thermal.clear();
	//int color_dim = PATCH_DIM - CHANNEL_DIM;
	for (int i = 0; i<color_dim; ++i)
		integ_hist_thermal.push_back(cv::Mat(image_thermal.rows + 1, image_thermal.cols + 1, CV_32SC1));
	for (int i = 0; i<CHANNEL_DIM; ++i)
		integ_hist_thermal.push_back(cv::Mat(image_thermal.rows + 1, image_thermal.cols + 1, CV_64FC1));

	feature_map_thermal.resize(image_thermal.rows*image_thermal.cols, Eigen::VectorXd::Zero(PATCH_DIM));

	// 全局的
	feature_map_bbox_thermal.resize(image_thermal.rows*image_thermal.cols, Eigen::VectorXd::Zero(BBOX_DIM)); // 提取全局的
	cv::split(image_thermal, image_channel_thermal);
	/****************** 加入热红外thermal *********************/

	compute_color_histogram_map();		//加入thermal
	compute_gradient_histogram_map();	//加入thermal
	compute_feature_map();				//加入thermal
}

// learning from weak and noisy labels for semantic segmentation
void soft_thresholding(Eigen::MatrixXd& p, Eigen::MatrixXd& temp, double lambda1)
{

	//std::cout << "soft_thresholding: " << std::endl;
	//for (int i = 0; i < p.rows(); i++)
	//{
	//	std::cout << temp(i, 0) << " ";
	//}
	//std::cout << "\n lambda1: " << lambda1 << std::endl;

	//temp ->B   lamda1-> lambda/2
	double x = 0.0f;
	double y = 0.0f;
	double z = 0.0f;
	for (int i = 0; i < p.rows(); i++)
	{
		//std::cout << temp(i, 0) << " ";
		if (temp(i, 0) < -lambda1)
		{
			z = temp(i, 0) + lambda1;
		}
		else if (temp(i, 0)>lambda1)
		{
			z = temp(i, 0) - lambda1;
		}
		else
		{
			z = 0;
		}
		p(i) = z;
	}
	/*std::cout << "\n p: " << std::endl;
	for (int i = 0; i < p.rows(); i++)
	{
		std::cout << p(i) << " ";
	}*/
	//system("pause");
}


// optimization: two L2-norm and one L1-norm
void soft_thresholding_three_terms(Eigen::MatrixXd& q, Eigen::MatrixXd& e1, Eigen::MatrixXd& e2, Eigen::MatrixXd& e3, Eigen::MatrixXd&/*double*/ xi_1, double xi_2, double xi_3)
{
	
	double x = 0.0f; // z_1 in the q-subproblem
	double y = 0.0f; // z_2 in the q-subproblem
	double z = 0.0f;
	//double z2 = 0;
	

	for (int i = 0; i < q.rows(); i++)
	{
		x = (xi_1(i,i) * e1(i) + xi_2 * e2(i)) / (xi_1(i,i) + xi_2);
		y = xi_3 / (2 * (xi_1(i,i) + xi_2));
		if (x - e3(i) < -y)
		{
			z = x + y;
		}
		else if (x - e3(i) <= y)
		{
			z = e3(i); 
		}
		else
		{
			z = x - y;
		}
		q(i) =z;
	}
}

void solve_l21(Eigen::MatrixXd& M, Eigen::MatrixXd& N, double sigma)
{
	int nr = M.rows();
	int nc = M.cols();

	for (int i = 0; i < nc; i++)
	{
		Eigen::MatrixXd x = N.col(i);
		double norm = x.norm();
		if (norm > sigma)
		{
			M.col(i) = (norm - sigma) * x / norm;
		}
		else
		{
			M.col(i).setZero();
		}
	}
}



std::vector<Eigen::MatrixXd> Tracker::manifold_ranking_denoise_graph(std::vector<Rect>& expanded_patch_V, std::vector<Rect>& expanded_patch_T, std::vector<Eigen::MatrixXd>& X, std::vector<Eigen::MatrixXd>& W, std::vector<Eigen::MatrixXd>& query, const int M)
{
	int nc = X[0].cols();

	Eigen::MatrixXd TempMatrix1 = Eigen::MatrixXd::Zero(nc, nc);
	Eigen::MatrixXd TempMatrix2 = Eigen::MatrixXd::Zero(nc, 1);

	std::vector<Eigen::MatrixXd> S;  // 学得的权重
	std::vector<Eigen::MatrixXd> S0; //优化后的种子点
	std::vector<Eigen::MatrixXd> D;
	std::vector<Eigen::MatrixXd> D_sqrt;
	std::vector<Eigen::MatrixXd> Q;
	std::vector<Eigen::MatrixXd> y2;
	std::vector<Eigen::MatrixXd> L;
	std::vector<Eigen::MatrixXd> H1;
	std::vector<Eigen::MatrixXd> H2;

	for (int i = 0; i < M; i++)
	{
		S.push_back(TempMatrix2);
		S0.push_back(TempMatrix2);
		Q.push_back(TempMatrix2);
		D.push_back(TempMatrix1);
		y2.push_back(TempMatrix2);
		H1.push_back(TempMatrix1);
		H2.push_back(TempMatrix2);
		L.push_back(TempMatrix1);
		D_sqrt.push_back(TempMatrix1);

	}


	Eigen::MatrixXd I = Eigen::MatrixXd::Zero(nc, nc);
	Eigen::MatrixXd Y1 = Eigen::MatrixXd::Zero(nc, 1);

	Eigen::MatrixXd SM = Eigen::MatrixXd::Zero(M*nc, 1);
	Eigen::MatrixXd pSM = Eigen::MatrixXd::Zero(M*nc, 1);
	Eigen::MatrixXd QM = Eigen::MatrixXd::Zero(M*nc, 1);
	Eigen::MatrixXd pQM = Eigen::MatrixXd::Zero(M*nc, 1);
	Eigen::MatrixXd S0M = Eigen::MatrixXd::Zero(M*nc, 1);
	Eigen::MatrixXd pS0M = Eigen::MatrixXd::Zero(M*nc, 1);
	Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nc, 1);
	Eigen::MatrixXd pP = Eigen::MatrixXd::Zero(nc, 1);

	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nc, M*nc);
	Eigen::MatrixXd LM = Eigen::MatrixXd::Zero(M*nc, M*nc);

	Eigen::MatrixXd D3 = Eigen::MatrixXd::Zero(nc, nc);
	Eigen::MatrixXd D3M = Eigen::MatrixXd::Zero(M*nc, M*nc);

	const double eta =(1.0 / M)*(X[0].squaredNorm() + X[1].squaredNorm());

	for (int i = 0; i < nc; i++)
	{
		I(i, i) = 1;
	}

	//C=(I,-I)
	for (int i = 0; i < nc; i++)
		for (int j = 0; j < 2 * nc; j++)
		{
			C(i, i) = 1;
			if (i == j % 100 && j / 100 != 0)
				C(i, j) = -1;
		}


	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			D[i](j, j) = W[i].col(j).sum();
		}
	}

	for (int i = 0; i < M; i++)
		for (int j = 0; j < nc; j++)
		{
			D_sqrt[i](j, j) = 1.0 / sqrt(D[i](j, j));
		}

	for (int i = 0; i < M; i++)
	{
		L[i] = I - D_sqrt[i] * W[i] * D_sqrt[i];
	}

	for (int ni = 0; ni < nc; ni++)
		for (int nj = 0; nj < nc; nj++)
		{
			LM(ni, nj) = L[0](ni, nj);
		}

	for (int ni = nc; ni < M*nc; ni++)
		for (int nj = nc; nj < M*nc; nj++)
		{
			LM(ni, nj) = L[1](ni-nc, nj-nc);
		}

	double max_err = 0;
	double pho = 1.5;
	double mu = 1e-3;
	const double max_mu = 1e10;
	const double tol = 1e-4;

	for (int i = 0; i < nc; i++)
	{
		if (query[0](i) == 1 || query[0](i) == -1)
		{
			D3(i, i) = 1;
		}
	}
	for (int i = 0; i < nc; i++)
		for (int j = 0; j < nc; j++)
		{
			D3M(i, j) = D3(i, j);
		}

	for (int i = nc; i < M*nc; i++)
		for (int j = nc; j < M*nc; j++)
		{
			D3M(i, j) = D3(i-nc, j-nc);
		}
	
	for (int i = 0; i < M; i++)
	{
		for (int ii = 0; ii < nc; ii++)
		{
			if (query[i](ii) == -1)
			{
				query[i](ii) = 0;
			}
		}
	}

	int num = 1;
	for (int i = 0; i < max_iter; i++)
	{

		//update qm
		for (int ii = 0; ii < M; ii++)
		{
			H1[ii] = 4 * ALPHA*(D[ii] - W[ii]) + mu*I;
			Q[ii] = H1[ii].inverse()*(mu*S0[ii] + y2[ii]);
		}

		for (int j = 0; j < M*nc; j++)
		{
			if (j >= 0 && j <= 99)
			{
				QM(j) = Q[0](j);
			}
			else
			{
				QM(j) = Q[1](j-100);
			}
		}


		//update s0m
		Eigen::MatrixXd tmp_f = LAMBDA*D3;

		for (int i = 0; i < M; i++)
		{
			H2[i] = Q[i] - y2[i] / mu;
		}

		soft_thresholding_three_terms(S0[0], S[0], H2[0], query[0], tmp_f, mu / 2, BETA);
		soft_thresholding_three_terms(S0[1], S[1], H2[1], query[1], tmp_f, mu / 2, BETA);

		for (int i = 0; i < M*nc; i++)
		{
			if (i <= 99 && i >= 0)
			{
				S0M(i) = S0[0](i);
			}
			else{
				S0M(i) = S0[1](i-100);
			}
		}

		//update p
		Eigen::MatrixXd temp;
		temp = C*pSM - Y1 / mu;
		//std::cout << "temp value: " << std::endl;
		/*for (int i = 0; i < nc; i++)
		{
			std::cout << temp(i, 0) << " ";
		}*/
		//std::cout << "\n lambda1 / mu: " << lambda1 / mu << std::endl;

		//soft_thresholding(P, tmp_f, lambda1 / mu);
		soft_thresholding(P, temp, lambda1 / mu);
		//solve_l21(P, tmp_f, lambda1 / mu);

		//update S
		Eigen::MatrixXd DELTAsfk,temp11,temp12,temp13;

		temp11 = 2 * LM*pSM;
		temp12 = 2 * LAMBDA*D3M*(pSM - pS0M);
		temp13 = mu*C.transpose()*(pP - temp);
		DELTAsfk = temp11 + temp12 - temp13 + lambda2*pSM;
		SM = pSM - DELTAsfk / (eta*mu);


		for (int i = 0; i < M*nc; i++)
		{
			SM(i) = SM(i) > 0 ? SM(i) : 0;
		}

		for (int i = 0; i < M*nc; i++)
		{
			if (i >= 0 && i <= 99)
			{
				S[0](i) = SM(i);
			}
			else
			{
				S[1](i-100) = SM(i);
			}
		}



		temp = QM - pQM;
		double max_err1 = temp.maxCoeff() + (-temp).maxCoeff();

		temp = S0M - pS0M;
		double max_err2 = temp.maxCoeff() + (-temp).maxCoeff();

		temp = P - pP;
		double max_err3 = temp.maxCoeff() + (-temp).maxCoeff();

		temp = SM - pSM;
		double max_err4 = temp.maxCoeff() + (-temp).maxCoeff();

		max_err = MAX(max_err1, max_err2);
		max_err = MAX(max_err, max_err3);
		max_err = MAX(max_err, max_err4);

		if (max_err < tol)
			break;

		mu = MIN(max_mu, mu*pho);

	//	std::cout << "num:" << num++ << " " << " max_err:" <<" "<< max_err << std::endl;
		Y1 = Y1 + mu*(P - C*S0M);
		for (int i = 0; i < M; i++)
		{
			y2[i] = y2[i] + mu*(S0[i] - Q[i]);
		}
		
		pQM = QM;
		pS0M = S0M;
		pP = P;
		pSM = SM;

		/*for (int i = 0; i < M - 1; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				std::cout << S[i](j) << " ";
				if ((j + 1) % 10 == 0)
					std::cout << std::endl;
			}
		}
		system("pause");*/

	}
	/*	for (int i = 0; i < M - 1; i++)
	{
	for (int j = 0; j < nc; j++)
	{
	std::cout << S[i](j) << " ";
	if ((j + 1) % 10 == 0)
	std::cout << std::endl;
	}
	}
	system("pause");*/
	return S;
	
}

void Tracker::update_patch_weight()
{
	std::vector<Rect> expanded_patch_V = extract_expanded_patch(object_bbox, expanded_patch_mask);
	std::vector<Eigen::VectorXd> expanded_feature_V(expanded_patch_V.size(), Eigen::VectorXd(OBJECT_DIM));// 每个patch的特征维度为PATCH_DIM(32);

	std::vector<Rect> expanded_patch_T = extract_expanded_patch(object_bbox, expanded_patch_mask_thermal);
	std::vector<Eigen::VectorXd> expanded_feature_T(expanded_patch_T.size(), Eigen::VectorXd(OBJECT_DIM));


	for (int i = 0; i<expanded_patch_V.size(); ++i)
	{
		expanded_feature_V[i] = extract_expanded_patch_feature(expanded_patch_V[i]);
		expanded_feature_T[i] = extract_expanded_patch_feature_thermal(expanded_patch_V[i]);
	}

	int rows = expanded_patch_V.size();
	Eigen::MatrixXd W_V = Eigen::MatrixXd::Zero(rows, rows);
	Eigen::MatrixXd W_T = Eigen::MatrixXd::Zero(rows, rows);
	// 
	Eigen::MatrixXd X_V = Eigen::MatrixXd::Zero(PATCH_DIM, expanded_patch_V.size());
	Eigen::MatrixXd X_T = Eigen::MatrixXd::Zero(PATCH_DIM, expanded_patch_T.size());	//新加入thermal

	int nc = expanded_patch_V.size();
	//int nc = expanded_patch.size() + 1; // 全局的，最后扩充一个整体的
	Eigen::VectorXd score_V;
	Eigen::VectorXd score_T;
	

	//std::vector<std::vector<Eigen::VectorXd> > X;
	std::vector<Eigen::MatrixXd> X;
	X.clear();						//下次使用前要清空容器

	std::vector<Eigen::MatrixXd> query;
	query.clear();

	std::vector<Eigen::MatrixXd> W;
	W.clear();


	std::vector<Eigen::MatrixXd> score;
	score.clear();

					//下次使用前要清空容器
	for (int i = 0; i < expanded_patch_V.size(); ++i)
	{
		X_V.col(i) = expanded_feature_V[i];
		X_T.col(i) = expanded_feature_T[i];	//新加入thermal

	}
	X.push_back(X_V);
	X.push_back(X_T);

	
	score.push_back(score_V);
	score.push_back(score_T);


	Eigen::MatrixXd query_V = Eigen::MatrixXd::Zero(nc, 1);
	Eigen::MatrixXd query_T = Eigen::MatrixXd::Zero(nc, 1);

	Eigen::MatrixXd I = Eigen::MatrixXd::Zero(nc, nc);

	Rect s_bbox(object_bbox.x + 0.1f*object_bbox.w,
		object_bbox.y + 0.1f*object_bbox.h,
		0.8f*object_bbox.w,
		0.8f*object_bbox.h);

	
	for (int i = 0; i < rows; ++i) {
		for (int j = i; j < rows; ++j) {
				double dist1 = abs(expanded_patch_V[i].x - expanded_patch_V[j].x) / expanded_patch_V[i].w;
				double dist2 = abs(expanded_patch_V[i].y - expanded_patch_V[j].y) / expanded_patch_V[i].h;

				if ((dist1 > NEIGHBOR_DIST_THRESHOLD || dist2 > NEIGHBOR_DIST_THRESHOLD))continue;

				double similarity = exp(-GAMMA*((expanded_feature_V[i] - expanded_feature_V[j]).squaredNorm()));//这是计算W的那个公式

				W_V(i, j) = similarity; //计算W
				W_V(j, i) = similarity; //计算W  
		}

			if (expanded_patch_V[i].is_inside(s_bbox)) {
				query_V(i) = 1;
			}
			else if (expanded_patch_V[i].is_inside(object_bbox)) {
				query_V(i) = 0;
			}
			else {
				query_V(i) = -1;
			}
	}


	for (int i = 0; i<rows; ++i) {
		for (int j = i; j < rows; ++j) {
				double dist1 = abs(expanded_patch_T[i].x - expanded_patch_T[j].x) / expanded_patch_T[i].w;
				double dist2 = abs(expanded_patch_T[i].y - expanded_patch_T[j].y) / expanded_patch_T[i].h;

				if ((dist1 > NEIGHBOR_DIST_THRESHOLD || dist2 > NEIGHBOR_DIST_THRESHOLD))continue;

				double similarity = exp(-GAMMA*((expanded_feature_T[i] - expanded_feature_T[j]).squaredNorm()));
				W_T(i, j) = similarity;
				W_T(j, i) = similarity;
		}
		if (expanded_patch_T[i].is_inside(s_bbox)) {
			query_T(i) = 1.0f;
		}
		else if (expanded_patch_T[i].is_inside(object_bbox)) {
			query_T(i) = 0.0f;
		}
		else {
			query_T(i) = -1.0f;
		}
	}

	query.push_back(query_V);
	query.push_back(query_T);

	W.push_back(W_V);
	W.push_back(W_T);

	//std::cout << "123213" << std::endl;
	const int M = 2;
	score = manifold_ranking_denoise_graph(expanded_patch_V, expanded_patch_T, X, W, query, M); // 0--S  1--R
	//score_T = manifold_ranking_denoise_graph(expanded_patch_thermal, X_thermal,W_thermal,query_T, M);




	//	for (int i = 1; i < M ; i++)
	//{
	//for (int j = 0; j < nc; j++)
	//{
	//std::cout << score[i](j) << " ";
	//if ((j + 1) % 10 == 0)
	//std::cout << std::endl;
	//}
	//}
	//system("pause");

	// S
	float sum_V;
	float sum_T;
	for (int i = 0; i < M; i++)
	{
		 sum_V = score[0].sum();
	     sum_T = score[1].sum();
	}
	

	for (int i = 0; i < expanded_patch_V.size(); ++i)
	{
		score[0](i) = score[0](i) / (sum_V + 1e-6);
		score[1](i) = score[0](i) / (sum_T + 1e-6);
	}

	//for (int i = 0; i < M - 1; i++)
	//{
	//	for (int j = 0; j < nc; j++)
	//	{
	//		std::cout << score[i](j) << " ";
	//		if ((j + 1) % 10 == 0)
	//			std::cout << std::endl;
	//	}
	//}
	//system("pause");
	//std::ofstream ofs("d:\\weight.txt");
	//std::ofstream ofs("D:\\test\\weight.txt");
	//std::ofstream ofs1("D:\\test\\weight1.txt");
	//Eigen::MatrixXd patch_weight_write = Eigen::MatrixXd::Zero(expanded_patch.size(), 1);
	int idx = 0;
	for (int i = 0; i<expanded_patch_V.size(); ++i)
	{
		if (expanded_patch_V[i].is_inside(object_bbox))
		{
			patch_weight_v[idx] = 1 / (1 + exp(-SIGMA * (gamma_v*score[0](i) /*+ gamma_i*score[1](i)*/)));
			patch_weight_i[idx] = 1 / (1 + exp(-SIGMA * (gamma_v*score[1](i) /*+ gamma_i*score[1](i)*/)));
			//std::cout << patch_weight[idx] << std::endl;
			++idx;
		}
		
	}
	//for (int i = 0; i < 64; i++)
	//{
	//	std::cout << patch_weight[i] << " ";
	//	if ((i + 1) % 10 == 0)
	//		std::cout << std::endl;
	//}
	//system("pause");

	X.clear();						//下次使用前要清空容器
	W.clear();
	query.clear();
	score.clear();
}

// Keep the image size same with each other
void Tracker::image_alignment(std::vector<char*> &frame_files, std::vector<char*> &frame_files_thermal, int frame_id, cv::Mat &image, cv::Mat &image_thermal)
{
	int w, h;
	cv::Mat image_src = cv::imread(frame_files[frame_id - 1], cv::IMREAD_COLOR);
	cv::Mat image_src_thermal = cv::imread(frame_files_thermal[frame_id - 1], cv::IMREAD_COLOR);

	// chose the big one be the base size
	w = image_src.cols >= image_src_thermal.cols ? image_src.cols : image_src_thermal.cols;
	h = image_src.rows >= image_src_thermal.rows ? image_src.rows : image_src_thermal.rows;

	image = cv::Mat(h, w, image_src.type()); // h,  w
	image_thermal = cv::Mat(h, w, image_src_thermal.type());

	for (int i = 0; i < image.rows; i++) // h
	{
		for (int j = 0; j < image.cols; j++)
		{
			image.at<cv::Vec3b>(i, j)[0] = image.at<cv::Vec3b>(i, j)[0]; // B
			image.at<cv::Vec3b>(i, j)[1] = image.at<cv::Vec3b>(i, j)[1]; // G
			image.at<cv::Vec3b>(i, j)[2] = image.at<cv::Vec3b>(i, j)[2]; // R
		}
	}

	for (int i = 0; i < image_thermal.rows; i++) // h
	{
		for (int j = 0; j < image_thermal.cols; j++)
		{
			image_thermal.at<cv::Vec3b>(i, j)[0] = image_thermal.at<cv::Vec3b>(i, j)[0]; // B
			image_thermal.at<cv::Vec3b>(i, j)[1] = image_thermal.at<cv::Vec3b>(i, j)[1]; // G
			image_thermal.at<cv::Vec3b>(i, j)[2] = image_thermal.at<cv::Vec3b>(i, j)[2]; // R
		}
	}

}

//void Tracker::compute_graph(Eigen::MatrixXd &W, std::vector<Rect> expanded_patch, std::vector<Eigen::VectorXd> expanded_feature, double thresold, char flag[][100], int init_point, int start_point)
//{
//	if (init_point >= 0 && init_point <= 99 && start_point >= 0 && start_point <= 99){
//		double temp = 0.0f;
//
//		int graph[8] = { start_point + 10, start_point + 11, start_point + 1, start_point - 9, start_point - 10, start_point - 11, start_point - 1, start_point + 9 };//找出邻域值下标
//
//		if (!flag[init_point][start_point] && (temp = exp(-GAMMA*((expanded_feature[init_point] - expanded_feature[start_point]).squaredNorm()))) > thresold) {
//			//std::cout << temp << std::endl;
//			W(init_point, start_point) = temp;
//			W(start_point, init_point) = W(init_point, start_point);
//			flag[start_point][init_point] = flag[init_point][start_point] = 1;
//		}
//		else {
//			return;
//		}
//
//		for (int i = 0; i < 8; i++)//计算八邻域 依次递归
//		{
//			if ((graph[i] <100 && graph[i] >= 0) && !(abs(expanded_patch[start_point].x - expanded_patch[graph[i]].x) / expanded_patch[start_point].w > NEIGHBOR_DIST_THRESHOLD || abs(expanded_patch[graph[i]].y - expanded_patch[start_point].y) / expanded_patch[start_point].h > NEIGHBOR_DIST_THRESHOLD))
//			{ //强迫为八邻域
//				if (!flag[start_point][graph[i]] && (temp = exp(-GAMMA*((expanded_feature[start_point] - expanded_feature[graph[i]]).squaredNorm()))) > thresold)
//				{
//					//std::cout << temp << std::endl;
//					//	W(start_point, xiao_biao[i]) = temp;
//					W(graph[i], start_point) = W(start_point, graph[i]);
//
//					if (start_point != init_point)
//						flag[graph[i]][start_point] = flag[start_point][graph[i]] = 1;
//
//					compute_graph(W, expanded_patch, expanded_feature, thresold, flag, init_point, graph[i]);
//				}
//			}
//		}
//	}
//}

//void solve_l21(Eigen::MatrixXd& M, Eigen::MatrixXd& N, double sigma)
//{
//	int nr = M.rows();
//	int nc = M.cols();
//
//	for (int i = 0; i < nc; i++)
//	{
//		Eigen::MatrixXd x = N.col(i);
//		double norm = x.norm();
//		if (norm > sigma)
//		{
//			M.col(i) = (norm - sigma) * x / norm;
//		}
//		else
//		{
//			M.col(i).setZero();
//		}
//	}
//}

/*将数组a[s]...a[t]中的元素用一个元素划开，保存中a[k]中*/
void partition1(Eigen::MatrixXd& a, int s, int t, int &k)
{
	int i, j;
	double x = a[s];    //取划分元素     
	i = s;        //扫描指针初值     
	j = t;
	do
	{
		while ((a[j]>x) && i<j) j--;   //从右向左扫描,如果是比划分元素大，则不动  
		if (i<j) a[i++] = a[j];           //小元素向左边移     
		while ((a[i] <= x) && i<j) i++;      //从左向右扫描，如果是比划分元素小，则不动   
		if (i<j) a[j--] = a[i];            //大元素向右边移     

	} while (i<j); //直到指针i与j相等      
	a[i] = x;      //划分元素就位     
	k = i;
}

int FindKMin(Eigen::MatrixXd& a, int low, int high, int k)
{
	int q;
	int index = -1;
	if (low < high)
	{
		partition1(a, low, high, q);
		int len = q - low + 1; //表示第几个位置      
		if (len == k)
			index = q; //返回第k个位置     
		else if (len < k)
			index = FindKMin(a, q + 1, high, k - len);
		else
			index = FindKMin(a, low, q - 1, k);
	}
	return index;

}


double MaxValue(double x, double y)
{
	int max = 0;
	max = y >= x ? y : x;
	return max;
}





void Tracker::update_classifier(bool is_first)
{
	std::vector<Rect> train_sample = extract_train_sample(object_bbox);
	std::vector<Eigen::VectorXd> train_features(train_sample.size(), Eigen::VectorXd(2 * OBJECT_DIM /*+ 2 * BBOX_DIM*/));//新添加的thermal特征，特征维度*2
	for (int i = 0; i<train_sample.size(); ++i)
		train_features[i] = extract_train_feature(train_sample[i]);

	classifier.train(train_sample, train_features, 0);
	if (is_first)
	{
		classifier0.train(train_sample, train_features, 0);
	}
}

//测试分类器时，提取多模态的特征，融合两个模态的特征
Eigen::VectorXd Tracker::extract_test_feature(const Rect &sample)
{
	//Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * OBJECT_DIM));//新加入thermal特征，特征维度*2
	Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * OBJECT_DIM + 2 * BBOX_DIM));//新加入thermal特征，特征维度*2
	//Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * BBOX_DIM));//新加入thermal特征，特征维度*2

	for (int i = 0; i<NUM_PATCH; ++i)
	{
		int x_min = sample.x + patch_mask[i].x;
		int y_min = sample.y + patch_mask[i].y;
	
		feature.segment(i*PATCH_DIM * 2, PATCH_DIM) = patch_weight_v[i] * feature_map[image.cols*y_min + x_min];
		feature.segment(i*PATCH_DIM * 2 + PATCH_DIM, PATCH_DIM) = patch_weight_i[i] * feature_map_thermal[image_thermal.cols*y_min + x_min];
		//std::cout << feature_map[image.cols*y_min + x_min] << std::endl;
	}

	Rect r_bbox(object_bbox.x, object_bbox.y, object_bbox.w, object_bbox.h);
	
	//feature.segment(0, BBOX_DIM) = global_ration*feature_map_bbox[image.cols*sample.y + sample.x];
	//feature.segment(0 + BBOX_DIM, BBOX_DIM) = global_ration*feature_map_bbox_thermal[image_thermal.cols*sample.h + sample.x];
	feature.segment(OBJECT_DIM * 2, BBOX_DIM) = global_ration*feature_map_bbox[image.cols*sample.y + sample.x - 1];// 
	feature.segment(OBJECT_DIM * 2 + BBOX_DIM, BBOX_DIM) = global_ration*feature_map_bbox_thermal[image_thermal.cols*sample.y + sample.x - 1];// 

	//std::cout << "1: " << feature_map_bbox[image.cols*sample.y + sample.x].transpose() << std::endl;
	//std::cout << "2: " << feature_map_bbox[image.cols*(sample.h) + sample.w].transpose() << std::endl;
	feature.normalize();
	return feature;
}

//训练分类器时，提取多模态特征，融合两个模态的特征
Eigen::VectorXd Tracker::extract_train_feature(const Rect &sample)
{
	//Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * OBJECT_DIM));//新加入thermal特征，特征维度*2
	Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * OBJECT_DIM + 2 * BBOX_DIM)); //将全局的特征加入进来
	////Eigen::VectorXd feature(Eigen::VectorXd::Zero(2 * BBOX_DIM)); //将全局的特征加入进来
	for (int j = 0; j<NUM_PATCH; ++j)
	{
		int x_min = sample.x + patch_mask[j].x;
		int y_min = sample.y + patch_mask[j].y;

		Rect r(x_min, y_min, patch_w, patch_h);
		if (r.is_inside(feature_bbox))
		{
			feature.segment(j*PATCH_DIM * 2, PATCH_DIM) = patch_weight_v[j] * feature_map[image.cols*y_min + x_min];	//两个模态共同训练出一个patch_weight[]
			feature.segment(j*PATCH_DIM * 2 + PATCH_DIM, PATCH_DIM) = patch_weight_i[j] * feature_map_thermal[image_thermal.cols*y_min + x_min];
			//feature.segment(j*PATCH_DIM * 2, PATCH_DIM) = (rank_foreground + rank_background) * patch_weight[j] * feature_map[image.cols*y_min + x_min];	//两个模态共同训练出一个patch_weight[]
			//feature.segment(j*PATCH_DIM * 2 + PATCH_DIM, PATCH_DIM) = (rank_foreground_thermal + rank_background_thermal) * patch_weight[j] * feature_map_thermal[image_thermal.cols*y_min + x_min];
		}
		else
		{
			feature.segment(j*PATCH_DIM * 2, PATCH_DIM) = patch_weight_v[j] * extract_patch_feature(r);	//两个模态共同训练出一个patch_weight[]
			feature.segment(j*PATCH_DIM * 2 + PATCH_DIM, PATCH_DIM) = patch_weight_i[j] * extract_patch_feature_thermal(r);
			//feature.segment(j*PATCH_DIM * 2, PATCH_DIM) = ((rank_foreground + rank_background)) * patch_weight[j] * extract_patch_feature(r);	//两个模态共同训练出一个patch_weight[]
			//feature.segment(j*PATCH_DIM * 2 + PATCH_DIM, PATCH_DIM) = (rank_foreground_thermal + rank_background_thermal)* patch_weight[j] * extract_patch_feature_thermal(r);
		}
	}
	//Rect r_bbox(object_bbox.x, object_bbox.y, object_bbox.w, object_bbox.h);
	Rect r_bbox(sample.x, sample.y, sample.w, sample.h);
	//feature.segment(0, BBOX_DIM) = global_ration * extract_bbox_feature(r_bbox);
	//feature.segment(0 + BBOX_DIM, BBOX_DIM) = global_ration * extract_bbox_feature_thermal(r_bbox);
	feature.segment(OBJECT_DIM * 2, BBOX_DIM) = global_ration * extract_bbox_feature(r_bbox);
	feature.segment(OBJECT_DIM * 2 + BBOX_DIM, BBOX_DIM) = global_ration * extract_bbox_feature_thermal(r_bbox);

	//std::cout << "train: " << extract_bbox_feature(r_bbox).transpose() << std::endl;

	feature.normalize();
	return feature;
}

Eigen::VectorXd Tracker::extract_patch_feature(const Rect &patch)
{
	int x_min = (int)patch.x;
	int y_min = (int)patch.y;
	int x_max = (int)(patch.x + patch.w);
	int y_max = (int)(patch.y + patch.h);

	return extract_patch_feature(x_min, y_min, x_max, y_max);
}

Eigen::VectorXd Tracker::extract_patch_feature(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature(PATCH_DIM);

	double total_sum = 0;
	int color_dim = PATCH_DIM - CHANNEL_DIM;
	double patch_area = patch_w*patch_h;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist[i].at<int>(y_min, x_min)
			+ integ_hist[i].at<int>(y_max, x_max)
			- integ_hist[i].at<int>(y_max, x_min)
			- integ_hist[i].at<int>(y_min, x_max);
		feature[i] = sum / patch_area;
		
		//total_sum += sum*sum;
		//feature[i] = sum*2.2;
	}

	//for (int i = 0; i<color_dim; ++i)
	//	feature[i] /= (sqrt(total_sum) + 1e-6);

	total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist[color_dim + i].at<double>(y_max, x_max)
			- integ_hist[color_dim + i].at<double>(y_max, x_min)
			- integ_hist[color_dim + i].at<double>(y_min, x_max);

		//total_sum += sum*sum;
		total_sum += sum;
		feature[color_dim + i] = sum*1.5;
	}
	for (int i = 0; i<grad_dim; ++i)
		//feature[color_dim + i] /= (sqrt(total_sum) + 1e-6);
		feature[color_dim + i] /= (total_sum + 1e-6);

	
	//std::cout << feature << std::endl;
	return feature;
}

Eigen::VectorXd Tracker::extract_expanded_patch_feature(const Rect &patch)
{
	Eigen::VectorXd feature;
	if (patch.is_inside(feature_bbox))
		feature = feature_map[image.cols*patch.y + patch.x];
	else
		feature = extract_patch_feature(patch);

	return feature;
}

Eigen::VectorXd Tracker::extract_patch_feature_thermal(const Rect &patch)
{
	int x_min = (int)patch.x;
	int y_min = (int)patch.y;
	int x_max = (int)(patch.x + patch.w);
	int y_max = (int)(patch.y + patch.h);

	return extract_patch_feature_thermal(x_min, y_min, x_max, y_max);
}

Eigen::VectorXd Tracker::extract_patch_feature_thermal(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature_thermal(PATCH_DIM);

	double total_sum = 0;
	int color_dim = PATCH_DIM - CHANNEL_DIM;
	double patch_area = patch_w*patch_h;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist_thermal[i].at<int>(y_min, x_min)
			+ integ_hist_thermal[i].at<int>(y_max, x_max)
			- integ_hist_thermal[i].at<int>(y_max, x_min)
			- integ_hist_thermal[i].at<int>(y_min, x_max);
		feature_thermal[i] = sum / patch_area;

		//total_sum += sum*sum;
		//feature[i] = sum*2.2;
	}

	//for (int i = 0; i<color_dim; ++i)
	//	feature[i] /= (sqrt(total_sum) + 1e-6);

	total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist_thermal[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist_thermal[color_dim + i].at<double>(y_max, x_max)
			- integ_hist_thermal[color_dim + i].at<double>(y_max, x_min)
			- integ_hist_thermal[color_dim + i].at<double>(y_min, x_max);

		//total_sum += sum*sum;
		total_sum += sum;
		feature_thermal[color_dim + i] = sum*1.5;
	}
	for (int i = 0; i<grad_dim; ++i)
		//feature[color_dim + i] /= (sqrt(total_sum) + 1e-6);
		feature_thermal[color_dim + i] /= (total_sum + 1e-6);


	//std::cout << feature << std::endl;
	return feature_thermal;
}

// 获取global的信息，通过提取整个bounding box的信息实现
Eigen::VectorXd Tracker::extract_bbox_feature(const Rect &bbox)
{
	int x_min = (int)bbox.x;
	int y_min = (int)bbox.y;
	int x_max = (int)(bbox.x + bbox.w);
	int y_max = (int)(bbox.y + bbox.h);

	return extract_bbox_feature(x_min, y_min, x_max, y_max);
}

Eigen::VectorXd Tracker::extract_bbox_feature(int x_min, int y_min, int x_max, int y_max)
{
	//Eigen::VectorXd feature(PATCH_DIM);
	Eigen::VectorXd feature(BBOX_DIM);

	double total_sum = 0;
	//int color_dim = PATCH_DIM - CHANNEL_DIM;
	int color_dim = BBOX_DIM - CHANNEL_DIM;
	//double patch_area = patch_w*patch_h;
	double bbox_area = bbox_h * bbox_w;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist[i].at<int>(y_min, x_min)
			+ integ_hist[i].at<int>(y_max, x_max)
			- integ_hist[i].at<int>(y_max, x_min)
			- integ_hist[i].at<int>(y_min, x_max);
		//feature[i] = sum / patch_area;
		feature[i] = sum / bbox_area;

		total_sum += sum*sum;
		//feature[i] = sum*2.2;
	}

	//for (int i = 0; i<color_dim; ++i)
	//	//feature[i] /= (total_sum + 1e-6);
	//	feature[i] /= (sqrt(total_sum) + 1e-6);

	total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist[color_dim + i].at<double>(y_max, x_max)
			- integ_hist[color_dim + i].at<double>(y_max, x_min)
			- integ_hist[color_dim + i].at<double>(y_min, x_max);

		//total_sum += sum*sum;
		total_sum += sum;
		feature[color_dim + i] = sum*1.5;
	}
	for (int i = 0; i<grad_dim; ++i)
		//feature[color_dim + i] /= (sqrt(total_sum) + 1e-6);
		feature[color_dim + i] /= (total_sum + 1e-6);


	//std::cout << feature << std::endl;
	return feature;
}

Eigen::VectorXd Tracker::extract_bbox_feature_thermal(const Rect &bbox)
{
	int x_min = (int)bbox.x;
	int y_min = (int)bbox.y;
	int x_max = (int)(bbox.x + bbox.w);
	int y_max = (int)(bbox.y + bbox.h);

	return extract_bbox_feature_thermal(x_min, y_min, x_max, y_max);
}

Eigen::VectorXd Tracker::extract_bbox_feature_thermal(int x_min, int y_min, int x_max, int y_max)
{
	Eigen::VectorXd feature_thermal(BBOX_DIM);

	double total_sum = 0;
	int color_dim = BBOX_DIM - CHANNEL_DIM;
	//double patch_area = patch_w*patch_h;
	double bbox_area = bbox_h * bbox_w;
	for (int i = 0; i<color_dim; ++i)
	{
		double sum = integ_hist_thermal[i].at<int>(y_min, x_min)
			+ integ_hist_thermal[i].at<int>(y_max, x_max)
			- integ_hist_thermal[i].at<int>(y_max, x_min)
			- integ_hist_thermal[i].at<int>(y_min, x_max);
		//feature_thermal[i] = sum / patch_area;
		feature_thermal[i] = sum / bbox_area;
		//total_sum += sum*sum;
		//feature[i] = sum*2.2;
	}

	//for (int i = 0; i<color_dim; ++i)
	//	feature[i] /= (sqrt(total_sum) + 1e-6);

	total_sum = 0;
	int grad_dim = CHANNEL_DIM;
	for (int i = 0; i<grad_dim; ++i)
	{
		double sum = integ_hist_thermal[color_dim + i].at<double>(y_min, x_min)
			+ integ_hist_thermal[color_dim + i].at<double>(y_max, x_max)
			- integ_hist_thermal[color_dim + i].at<double>(y_max, x_min)
			- integ_hist_thermal[color_dim + i].at<double>(y_min, x_max);

		//total_sum += sum*sum;
		total_sum += sum;
		feature_thermal[color_dim + i] = sum*1.5;
	}
	for (int i = 0; i<grad_dim; ++i)
		//feature[color_dim + i] /= (sqrt(total_sum) + 1e-6);
		feature_thermal[color_dim + i] /= (total_sum + 1e-6);


	//std::cout << feature << std::endl;
	return feature_thermal;
}

Eigen::VectorXd Tracker::extract_expanded_patch_feature_thermal(const Rect &patch)
{
	Eigen::VectorXd feature_thermal;

	if (patch.is_inside(feature_bbox))
		feature_thermal = feature_map_thermal[image_thermal.cols*patch.y + patch.x];
	else
		feature_thermal = extract_patch_feature_thermal(patch);

	return feature_thermal;
}

void Tracker::compute_color_histogram_map()
{
	double bin_size = 32.0;
	int num_color_channel = NUM_CHANNEL - 1;

	for (int i = 0; i<num_color_channel; ++i)
	{
		cv::Mat tmp(image.rows, image.cols, CV_8UC1);
		tmp.setTo(0);

		for (int j = 0; j<CHANNEL_DIM; ++j)
		{
			for (int y = 0; y<image.rows; ++y)
			{
				const uchar* src = image_channel[i].ptr(y);
				uchar* dst = tmp.ptr(y);
				for (int x = 0; x<image.cols; ++x)
				{
					int bin = (int)((double)(*src) / bin_size);
					*dst = (bin == j) ? 1 : 0;
					++src;
					++dst;
				}
			}

			cv::integral(tmp, integ_hist[i*CHANNEL_DIM + j]);
		}

		/************** 新加入的热红外thermal **********/
		cv::Mat tmp_thermal(image_thermal.rows, image_thermal.cols, CV_8UC1);
		tmp_thermal.setTo(0);
		for (int j = 0; j<CHANNEL_DIM; ++j)
		{
			for (int y = 0; y<image_thermal.rows; ++y)
			{
				const uchar* src_thermal = image_channel_thermal[i].ptr(y);
				uchar*       dst_thermal = tmp_thermal.ptr(y);

				for (int x = 0; x<image_thermal.cols; ++x)
				{
					int bin_thermal = (int)((double)(*src_thermal) / bin_size);
					*dst_thermal = (bin_thermal == j) ? 1 : 0;
					++src_thermal;
					++dst_thermal;
				}
			}
			cv::integral(tmp_thermal, integ_hist_thermal[i*CHANNEL_DIM + j]);
		}
		/************** 新加入的热红外thermal **********/
	}
}

void Tracker::compute_gradient_histogram_map()
{
	float bin_size = 22.5;
	float radian_to_degree = 180.0 / CV_PI;

	std::vector<cv::Mat> bins;
	for (int i = 0; i<CHANNEL_DIM; ++i)
		bins.push_back(cv::Mat::zeros(image.rows, image.cols, CV_32FC1));

	std::vector<cv::Mat> x_sobel(image_channel.size()), y_sobel(image_channel.size());
	for (int k = 0; k < image_channel.size(); k++)	// 修改：取全部3通道
	{
		cv::Sobel(image_channel[k], x_sobel[k], CV_32FC1, 1, 0);
		cv::Sobel(image_channel[k], y_sobel[k], CV_32FC1, 0, 1);
	}
	
	float** x_sobel_row_ptr = new float*[image_channel.size()];
	float** y_sobel_row_ptr = new float*[image_channel.size()];
	for (int y = 0; y<image.rows; ++y)
	{
		for (int k = 0; k < image_channel.size(); k++)
		{
			x_sobel_row_ptr[k] = (float*)(x_sobel[k].row(y).data);
			y_sobel_row_ptr[k] = (float*)(y_sobel[k].row(y).data);
		}
		
		std::vector<float*> bins_row_ptrs(CHANNEL_DIM, nullptr);

		for (int i = 0; i<CHANNEL_DIM; ++i)
			bins_row_ptrs[i] = (float*)(bins[i].row(y).data);

		for (int x = 0; x<image.cols; ++x)
		{
			float magnitude = 0;
			int idx = 0;
			for (int k = 0; k < image_channel.size(); k++)
			{
				float temp = sqrt(x_sobel_row_ptr[k][x] * x_sobel_row_ptr[k][x] + y_sobel_row_ptr[k][x] * y_sobel_row_ptr[k][x]);
				if (temp > magnitude)
				{
					magnitude = temp;
					idx = k;
				}
			}
			

			if (x_sobel_row_ptr[idx][x] == 0)
				x_sobel_row_ptr[idx][x] += 0.00001;

			float orientation = atan(y_sobel_row_ptr[idx][x] / x_sobel_row_ptr[idx][x])*radian_to_degree + 90;
			

			for (int i = 1; i<CHANNEL_DIM; ++i)
			{
				if (orientation <= bin_size*i)
				{
					bins_row_ptrs[i - 1][x] = magnitude;
					break;
				}
			}
		}
	}
	delete[] x_sobel_row_ptr;
	delete[] y_sobel_row_ptr;

	/*cv::Mat gray_image(image.rows, image.cols, CV_8UC1);
	if (IMAGE_TYPE == cv::IMREAD_COLOR)
		cv::cvtColor(image, gray_image, CV_BGR2GRAY);
	else
		image.copyTo(gray_image);

	cv::Mat x_sobel, y_sobel;
	cv::Sobel(gray_image, x_sobel, CV_32FC1, 1, 0);
	cv::Sobel(gray_image, y_sobel, CV_32FC1, 0, 1);

	std::vector<cv::Mat> bins;
	for (int i = 0; i<CHANNEL_DIM; ++i)
		bins.push_back(cv::Mat::zeros(image.rows, image.cols, CV_32FC1));

	for (int y = 0; y<image.rows; ++y)
	{
		float* x_sobel_row_ptr = (float*)(x_sobel.row(y).data);
		float* y_sobel_row_ptr = (float*)(y_sobel.row(y).data);

		std::vector<float*> bins_row_ptrs(CHANNEL_DIM, nullptr);
		for (int i = 0; i<CHANNEL_DIM; ++i)
			bins_row_ptrs[i] = (float*)(bins[i].row(y).data);

		for (int x = 0; x<image.cols; ++x)
		{
			if (x_sobel_row_ptr[x] == 0)
				x_sobel_row_ptr[x] += 0.00001;

			float orientation = atan(y_sobel_row_ptr[x] / x_sobel_row_ptr[x])*radian_to_degree + 90;
			float magnitude = sqrt(x_sobel_row_ptr[x] * x_sobel_row_ptr[x] + y_sobel_row_ptr[x] * y_sobel_row_ptr[x]);

			for (int i = 1; i<CHANNEL_DIM; ++i)
			{
				if (orientation <= bin_size*i)
				{
					bins_row_ptrs[i - 1][x] = magnitude;
					break;
				}
			}
		}
	}*/

	/************** 新加入的thermal **************/
	std::vector<cv::Mat> bins_thermal;
	for (int i = 0; i<CHANNEL_DIM; ++i)
		bins_thermal.push_back(cv::Mat::zeros(image_thermal.rows, image_thermal.cols, CV_32FC1));

	std::vector<cv::Mat> x_sobel_thermal(image_channel_thermal.size());//, y_sobel_thermal(image_channel_thermal.size());
	std::vector<cv::Mat> y_sobel_thermal(image_channel_thermal.size());
	for (int k = 0; k < image_channel_thermal.size(); k++)	// 修改：取全部3通道
	{
		cv::Sobel(image_channel_thermal[k], x_sobel_thermal[k], CV_32FC1, 1, 0);
		cv::Sobel(image_channel_thermal[k], y_sobel_thermal[k], CV_32FC1, 0, 1);
	}

	float** x_sobel_row_ptr_thermal = new float*[image_channel_thermal.size()];
	float** y_sobel_row_ptr_thermal = new float*[image_channel_thermal.size()];
	for (int y = 0; y<image_thermal.rows; ++y)
	{
		for (int k = 0; k < image_channel_thermal.size(); k++)
		{
			x_sobel_row_ptr_thermal[k] = (float*)(x_sobel_thermal[k].row(y).data);
			y_sobel_row_ptr_thermal[k] = (float*)(y_sobel_thermal[k].row(y).data);
		}

		std::vector<float*> bins_row_ptrs_thermal(CHANNEL_DIM, nullptr);

		for (int i = 0; i<CHANNEL_DIM; ++i)
			bins_row_ptrs_thermal[i] = (float*)(bins_thermal[i].row(y).data);

		for (int x = 0; x<image_thermal.cols; ++x)
		{
			float magnitude = 0;
			int idx = 0;
			for (int k = 0; k < image_channel_thermal.size(); k++)
			{
				float temp = sqrt(x_sobel_row_ptr_thermal[k][x] * x_sobel_row_ptr_thermal[k][x] + y_sobel_row_ptr_thermal[k][x] * y_sobel_row_ptr_thermal[k][x]);
				if (temp > magnitude)
				{
					magnitude = temp;
					idx = k;
				}
			}


			if (x_sobel_row_ptr_thermal[idx][x] == 0)
				x_sobel_row_ptr_thermal[idx][x] += 0.00001;

			float orientation = atan(y_sobel_row_ptr_thermal[idx][x] / x_sobel_row_ptr_thermal[idx][x])*radian_to_degree + 90;


			for (int i = 1; i<CHANNEL_DIM; ++i)
			{
				if (orientation <= bin_size*i)
				{
					bins_row_ptrs_thermal[i - 1][x] = magnitude;
					break;
				}
			}
		}
	}
	delete[] x_sobel_row_ptr_thermal;
	delete[] y_sobel_row_ptr_thermal;
	/************** 新加入的thermal **************/

	int color_dim = PATCH_DIM - CHANNEL_DIM;
	for (int i = 0; i<CHANNEL_DIM; ++i)
		cv::integral(bins[i], integ_hist[color_dim + i]);

	/************** 新加入的thermal **************/
	for (int i = 0; i<CHANNEL_DIM; ++i)
		cv::integral(bins_thermal[i], integ_hist_thermal[color_dim + i]);
	/************** 新加入的thermal **************/
}

void Tracker::compute_feature_map()
{
	int x_start = -search_r;
	int x_end = search_r + object_bbox.w;
	int y_start = -search_r;
	int y_end = search_r + object_bbox.h;

	feature_bbox.set(object_bbox.x + x_start, object_bbox.y + y_start, x_end - x_start, y_end - y_start);
	for (int iy = y_start; iy <= y_end; ++iy)
	{
		int y_min = (int)object_bbox.y + iy;
		int y_max = y_min + patch_h;

		if ((y_min < border_bbox.y) || (y_max > border_bbox.y + border_bbox.h)) { continue; }
		double b = 0;
		for (int ix = x_start; ix <= x_end; ++ix)
		{
			int x_min = (int)object_bbox.x + ix;
			int x_max = x_min + patch_w;
			if ((x_min < border_bbox.x) || (x_max > border_bbox.x + border_bbox.w)) { continue; }

			feature_map[image.cols*y_min + x_min] = extract_patch_feature(x_min, y_min, x_max, y_max);
			feature_map_thermal[image_thermal.cols*y_min + x_min] = extract_patch_feature_thermal(x_min, y_min, x_max, y_max);
		}
	}
	// 提取全局的
	for (int iy = y_start; iy <= y_end; ++iy)
	{
		int y_min = (int)object_bbox.y + iy;
		int y_max = y_min + bbox_h;

		if ((y_min < border_bbox.y) || (y_max > border_bbox.y + border_bbox.h)) { continue; }
		double b = 0;
		for (int ix = x_start; ix <= x_end; ++ix)
		{
			int x_min = (int)object_bbox.x + ix;
			int x_max = x_min + bbox_w;
			if ((x_min < border_bbox.x) || (x_max > border_bbox.x + border_bbox.w)) { continue; }

			// 不能直接用extract_patch_feature, 里面有计算patch_area的，或者通过加一个flag(标记是patch还是bbox)实现
			feature_map_bbox[image.cols*y_min + x_min] = extract_bbox_feature(x_min, y_min, x_max, y_max); 
			feature_map_bbox_thermal[image_thermal.cols*y_min + x_min] = extract_bbox_feature_thermal(x_min, y_min, x_max, y_max);
		}
	}
}

std::vector<Rect> Tracker::extract_patch(Rect sample)
{
	std::vector<Rect> patch;

	int x_counter = (int)(sample.w / (double)patch_w);
	int y_counter = (int)(sample.h / (double)patch_h);

	int dx = (int)(0.5*(sample.w - x_counter*patch_w));
	int dy = (int)(0.5*(sample.h - y_counter*patch_h));

	int x_start = (int)sample.x;
	int y_start = (int)sample.y;

	for (int iy = 0; iy<y_counter; ++iy)
	{
		int y = y_start + iy*(patch_h);
		for (int ix = 0; ix<x_counter; ++ix)
		{
			int x = x_start + ix*(patch_w);
			patch.push_back(Rect(x, y, patch_w, patch_h));
		}
	}

	return patch;
}

std::vector<Rect> Tracker::extract_patch(Rect center, std::vector<Rect> mask)
{
	std::vector<Rect> samples;
	for (int i = 0; i<mask.size(); ++i)
	{
		int px_min = (int)(center.x + mask[i].x);
		int py_min = (int)(center.y + mask[i].y);
		int px_max = (int)(px_min + mask[i].w);
		int py_max = (int)(py_min + mask[i].h);

		Rect sample((float)px_min, (float)py_min, mask[i].w, mask[i].h);
		samples.push_back(sample);
	}

	return samples;
}

std::vector<Rect> Tracker::extract_expanded_patch(Rect sample)
{
	std::vector<Rect> expanded_patch;

	int x_counter = (int)(sample.w / (double)patch_w);
	int y_counter = (int)(sample.h / (double)patch_h);

	int x_start = (int)sample.x;
	int y_start = (int)sample.y;

	for (int iy = -1; iy<y_counter + 1; ++iy)
	{
		int y = y_start + iy*(patch_h);
		for (int ix = -1; ix<x_counter + 1; ++ix)
		{
			int x = x_start + ix*(patch_w);
			expanded_patch.push_back(Rect(x, y, patch_w, patch_h));
		}
	}

	return expanded_patch;
}

std::vector<Rect> Tracker::extract_expanded_patch(Rect center, std::vector<Rect> expanded_mask)
{
	std::vector<Rect> expanded_patch;
	for (int i = 0; i<expanded_mask.size(); ++i)
	{
		int px_min = (int)(center.x + expanded_mask[i].x);
		int py_min = (int)(center.y + expanded_mask[i].y);

		Rect patch(px_min, py_min, expanded_mask[i].w, expanded_mask[i].h);

		if (patch.is_inside(border_bbox))
			expanded_patch.push_back(patch);
	}

	return expanded_patch;
}

std::vector<Rect> Tracker::extract_train_sample(Rect center)
{
	int num_r = 5;
	int num_t = 16;
	double radius = 2 * search_r;
	double rstep = radius / 5.0;
	double tstep = 2.0*CV_PI / 16.0;

	std::vector<Rect> train_sample;
	train_sample.push_back(center);
	for (int ir = 1; ir <= num_r; ++ir)
	{
		double phase = (ir % 2)*tstep / 2;
		for (int it = 0; it<num_t; ++it)
		{
			double dx = ir*rstep*cosf(it*tstep + phase);
			double dy = ir*rstep*sinf(it*tstep + phase);

			Rect sample(center);
			sample.x = center.x + dx;
			sample.y = center.y + dy;

			if (sample.is_inside(image_bbox))
				train_sample.push_back(sample);
		}
	}

	return train_sample;
}