#pragma once

#include <vector>
#include <string>
#include <queue>
#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>

#include "Rect.h"
#include "Config.h"
#include "StructuredSVM.h"

class Tracker
{   
  const int NUM_PATCH = 64;
  const int CHANNEL_DIM = 8;

  const double SIGMA = 35; // scaling parameter for patch weight s, alpha in SOWP
  const double GAMMA = 5; // scaling parameter for edge weight W_(i,j)
  const double NEIGHBOR_DIST_THRESHOLD = 1.1;// 1.10;
 
  char flag[100][100];
  char flag_thermal[100][100];
  int NUM_MODALITY = 2; // We use RGB-T, two modality images
  
 /// const double NODES_CONNECT_THRESHOLD = 0.05;// 0.05; //gou_tu_threshold

  ///CVPR 2018 parameters start
  const double OMEGA = 0.4;// 396;// 396;// 0.396;//0.32; //thfirst frame information  **************************************************************
  const double THETA = 0.3;// the update threshold;    **************************************************************
  const double MAXITER = 50; //50 itermax
  const double max_iter = 50;
  const double ALPHA = 0.66; //0.7 图传递给降噪之后的种子点系数
  const double BETA = 0.9;// 0.023; //0.06 降噪系数BEAT*(S0-query)
  const double LAMBDA = 0.5;//0.01; 学权重同优化后的种子点的拟合系数LAMBDA*(S-S0)
  const double lambda1 = 0.9;	//跨模态约束系数lambda1*(Sm-Sm-1)
  const double lambda2 = 0.1;	//S的正则化系数lambda2*||S||
 // const double ETA_q = 0.5;// 0.5 // fitness

  const int numScales = 33;// 33;// 33->33  ******
  const double Scale_step = 1.054;// 1.056;//1.02->1.03   **
  const int interval = 3;//3->4   *****
  const double IsScale_Threshold = 0.0;	//0.0->0.0

  const double global_ration = 1.0;

  const double gamma_v = 1.0;
  const double gamma_i = 1.0;
  ///CVPR 2018 parameters end


  //const double BETA1 = 0.12;
  //const double MU = 0.92; // 0.9;// 1.2;//0.7; //alpha
 

  //////////////////para of A
  //const double delta = 11;		// affinity
 // const double gama = 0.1;		// sparse representation
 // const double lambda = 0.1;	// sparse noise
 // const double omega = 1;		//overfitting
 // const double XI = 6;		// 6 neighbor numbers               可调


 // const double lambda3 = 1.0;//1

  //const double PeakThreshold = 0.96;

  /*********************** 尺度处理 *********************/
  int frame_idx;
  bool is_scale;
  int scale_model_id;
  Rect best_bbox;
  Rect object_temp_bbox;

  cv::Mat image_s;
  std::vector<Eigen::VectorXd> feature_map_s;
  std::vector<cv::Mat> image_channel_s;
  std::vector<cv::Mat> integ_hist_s;

  /********* 尺度处理：新加入thermal ***********/
  cv::Mat image_thermal_s;
  std::vector<Eigen::VectorXd> feature_map_thermal_s;
  std::vector<cv::Mat> image_channel_thermal_s;
  std::vector<cv::Mat> integ_hist_thermal_s;
  /********* 尺度处理：新加入thermal ***********/

  std::vector<Rect> best_sample_s;
  std::vector<double> validation_score_s;
  double currentScaleFactor = 1;
  double min_scale_factor;
  double max_scale_factor;
  std::vector<double> scaleFactors;//save scale models
  std::vector<double> scale_window;//hann(33)
  /*********************** 尺度处理 *********************/

  std::vector<float> iter_errs;
  std::vector<float> mu_errs;

  const int IMAGE_TYPE;
  const int IMAGE_TYPE_THERMAL;//新加入的thermal
  const int INIT_FRAME;
  const int END_FRAME;
  const int NUM_FRAME;
  const int NUM_CHANNEL;    
  const int PATCH_DIM;
  const int OBJECT_DIM;
  const std::string SEQUENCE_NAME;
  const std::string SEQUENCE_PATH;
  const std::string RESULT_PATH;
  const Rect INIT_BOX;

  const int BBOX_DIM;
  /*****************自己加的热红外*******************/
  const std::string SEQUENCE_PATH_THERMAL;
  /*****************自己加的热红外*******************/

public:
  Tracker(Config &config);
  ~Tracker();  
  void run();  
  void save(std::string cur_path);

private:  
  void track(int t);
  void initialize();
  void initialize_bbox(); 
  void initialize_mask();
  void initialize_seed();
  
  bool update_object_box(int frame_id);
  void update_feature_map(int frame_id);  
  void update_patch_weight();
  void update_classifier(bool is_first = false);
  void update_result_box(int frame_id);

  void compute_color_histogram_map();
  void compute_gradient_histogram_map();
  void compute_feature_map();

  void compute_graph(Eigen::MatrixXd &W, std::vector<Rect> expanded_patch, std::vector<Eigen::VectorXd> expanded_feature, double thresold, char flag[][100], int init_point, int start_point);
  Eigen::MatrixXd Tracker::ConstructWMatrix(std::vector<Rect> expanded_patch, std::vector<Eigen::MatrixXd>& X);

  std::vector<Eigen::MatrixXd> manifold_ranking_denoise_graph(std::vector<Rect>& expanded_patch_V, std::vector<Rect>& expanded_patch_T,/*std::vector<std::vector<Eigen::VectorXd> >*/std::vector<Eigen::MatrixXd>& X, std::vector<Eigen::MatrixXd>& W, std::vector<Eigen::MatrixXd>& query, const int M);
  void Tracker::image_alignment(std::vector<char*> &frame_files, std::vector<char*> &frame_files_thermal, int frame_id, cv::Mat &image, cv::Mat &image_thermal);

  //Eigen::MatrixXd low_rank_sparse_weighted_graph(Eigen::MatrixXd& X, Eigen::MatrixXd& query);
  //Eigen::MatrixXd sparse_weighted_collaborative_graph(std::vector<Eigen::MatrixXd>& X, Eigen::MatrixXd& query, const int M);

  /****************尺度处理函数****************/
  void scale_estimation(int frame_id, const Rect &sample);
  Eigen::VectorXd extract_patch_feature_s(int x_min, int y_min, int x_max, int y_max);
  Eigen::VectorXd extract_patch_feature_thermal_s(int x_min, int y_min, int x_max, int y_max);		//新加入thermal项
  Eigen::VectorXd extract_test_feature_s(cv::Mat &expand_roi, cv::Mat &expand_roi_thermal);			//新加入thermal项
  void extract_sample_feature_s(cv::Mat &roi, cv::Mat &expand_roi_thermal);							//新加入thermal项
  void compute_color_histogram_map_s(cv::Mat &roi, cv::Mat &expand_roi_thermal);					//新加入thermal项
  void compute_gradient_histogram_map_s(cv::Mat &roi, cv::Mat &expand_roi_thermal);					//新加入thermal项
  void compute_feature_map_s(cv::Mat &roi, cv::Mat &expand_roi_thermal);							//新加入thermal项
  void initial_scale_model();
  void update_scale();
  /****************尺度处理函数****************/

  /********************全局信息************************/
  std::vector<Eigen::VectorXd> feature_map_bbox; //提取全局
  std::vector<Eigen::VectorXd> feature_map_bbox_thermal;
  std::vector<Eigen::VectorXd> feature_map_bbox_s; //提取全局
  std::vector<Eigen::VectorXd> feature_map_bbox_thermal_s;

  Eigen::VectorXd extract_bbox_feature(const Rect &bbox);   // 获取global的信息，通过提取整个bounding box的信息实现
  Eigen::VectorXd extract_bbox_feature(int x_min, int y_min, int x_max, int y_max);
  Eigen::VectorXd extract_bbox_feature_thermal(const Rect &bbox);  // 获取global的信息，通过提取整个bounding box的信息实现
  Eigen::VectorXd extract_bbox_feature_thermal(int x_min, int y_min, int x_max, int y_max);

  Eigen::VectorXd extract_bbox_feature_s(int x_min, int y_min, int x_max, int y_max);
  Eigen::VectorXd extract_bbox_feature_thermal_s(int x_min, int y_min, int x_max, int y_max);		//新加入thermal项
  /********************全局信息************************/

  /**************** 多峰检测 ****************/
  Eigen::MatrixXd getMarkMatrix(Eigen::MatrixXd scoreMap, int h, int w);
  bool isTruePeak(Eigen::MatrixXd scoreMap, double, int index_x, int index_y);
  bool isOverBoundry(int index_x, int index_y, int x_shift, int yshift);
  /**************** 多峰检测 ****************/

  Eigen::VectorXd extract_test_feature(const Rect &sample);							//新加入thermal项
  Eigen::VectorXd extract_train_feature(const Rect &sample);						//新加入thermal项
  Eigen::VectorXd extract_patch_feature(const Rect &patch);
  Eigen::VectorXd extract_patch_feature(int x_min, int y_min, int x_max, int y_max);
  Eigen::VectorXd extract_expanded_patch_feature(const Rect &epatch);

  /**************** 新加入的thermal ****************/
  Eigen::VectorXd extract_patch_feature_thermal(const Rect &patch);
  Eigen::VectorXd extract_patch_feature_thermal(int x_min, int y_min, int x_max, int y_max);
  Eigen::VectorXd extract_expanded_patch_feature_thermal(const Rect &patch);
  /**************** 新加入的thermal ****************/

	std::vector<Rect> extract_patch(Rect center);
  std::vector<Rect> extract_patch(Rect center, std::vector<Rect> mask);
	std::vector<Rect> extract_expanded_patch(Rect center);
  std::vector<Rect> extract_expanded_patch(Rect center, std::vector<Rect> expanded_mask);
	std::vector<Rect> extract_train_sample(Rect center);

  void display_result(int t, std::string window_name, std::string window_name_thermal);


private:
  int patch_w;
  int patch_h;
  double scale_w;
  double scale_h;
  int search_r;

  int bbox_w; //提取全部的信息
  int bbox_h;

  std::vector<Eigen::MatrixXd> query_each_seq;
  std::vector<Eigen::MatrixXd> Q_each_seq;
  //std::vector<std::vector<double>> weight_each_seq;
  std::vector<Eigen::MatrixXd> weight_each_seq;
  std::vector<Eigen::MatrixXd> W_each_seq;
  std::vector<Eigen::MatrixXd> W_each_seq_thermal;
  std::vector<Eigen::MatrixXd> R_each_seq;
  std::vector<Eigen::MatrixXd> S_each_seq;

  std::vector<double> patch_weight_v;
  std::vector<double> patch_weight_i;
  std::vector<double> patch_weight0;
  std::vector<double> noise_norm;

  double rank_foreground;
  double rank_background;
  double rank_foreground_thermal;
  double rank_background_thermal;

  Eigen::VectorXd prev_fore_prob;
  Eigen::VectorXd prev_back_prob;

  //std::vector<double> R; // modality weight R[0]--visible  R[1]--thermal
  //std::vector<double> R_NEXT; // itera

  Rect image_bbox;
  Rect border_bbox;
  Rect feature_bbox;
  Rect object_bbox;  

  StructuredSVM classifier;
  StructuredSVM classifier0;
  std::vector<Eigen::VectorXd> feature_map;

  std::vector<Rect> result_box;
  std::vector<Rect> result_box_thermal;//可注释掉

  std::vector<Rect> patch_mask;
  std::vector<Rect> expanded_patch_mask;

  std::vector<Rect> patch_mask_thermal;
  std::vector<Rect> expanded_patch_mask_thermal;
  Rect bbox_mask; //提取全局的

  cv::Mat image;
  std::vector<cv::Mat> image_channel;
  std::vector<cv::Mat> integ_hist; 

  /***********自己添加的表示热红外图像********/
  std::vector<Eigen::VectorXd> feature_map_thermal;
  cv::Mat image_thermal;
  std::vector<cv::Mat> image_channel_thermal;
  std::vector<cv::Mat> integ_hist_thermal;
  /***********自己添加的表示热红外图像********/

  //std::vector<int> expanded_patch_corners;
};
