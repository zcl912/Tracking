#ifndef StructuredSVM_H
#define StructuredSVM_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "Rect.h"

struct SupportPattern
{
	std::vector<Eigen::VectorXd> x;
	std::vector<Rect> yv;
  std::vector<cv::Mat> img;
	int y;
	int refCount;
};

struct SupportVector
{
	SupportPattern* x;  
	int y;
	double b;
	double g;
};

class StructuredSVM
{
  const int ITER = 10;
  const double C = 10;
  const int MAX_BUDGET = 100;
public:
  StructuredSVM();
  ~StructuredSVM();

  double test(const Eigen::VectorXd &feature);
  double validation_test(const Eigen::VectorXd &feature);
  void train(const std::vector<Rect> &samples, const std::vector<Eigen::VectorXd> &features, int y);

private:
  Eigen::MatrixXd K;
  Eigen::VectorXd W;

  std::vector<SupportPattern*> sps;
  std::vector<SupportVector*> svs; 
  
	inline double Loss(const Rect& y1, const Rect& y2) const
	{
    return 1.0-y1.overlap_ratio(y2);
	}

	double ComputeDual() const;
	void SMOStep(int ipos, int ineg);	
  std::pair<int, double> MinGradient(int ind);

	void ProcessNew(int ind);
	void Reprocess();
	void ProcessOld();
	void Optimize();

	int AddSupportVector(SupportPattern* x, int y, double g);
	void RemoveSupportVector(int ind);
	void RemoveSupportVectors(int ind1, int ind2);
	void SwapSupportVectors(int ind1, int ind2);
	
	void BudgetMaintenance();
	void BudgetMaintenanceRemove();

  double evaluate(const Eigen::VectorXd& x) const;
  inline double compute_kernel_score(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
  {
    return x1.dot(x2);
  };
  inline double compute_kernel_score(const Eigen::VectorXd& x) const
  {
    return x.squaredNorm();
  };
};

#endif