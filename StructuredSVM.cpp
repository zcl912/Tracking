#include "stdafx.h"
#include <opencv2/imgproc/imgproc.hpp>

#include "StructuredSVM.h"

StructuredSVM::StructuredSVM() : K(Eigen::MatrixXd::Zero(MAX_BUDGET+2, MAX_BUDGET+2)) {}

StructuredSVM::~StructuredSVM()
{
  for (int i=0; i<sps.size(); ++i)
    delete sps[i];
  for (int i=0; i<svs.size(); ++i)
    delete svs[i];
}

double StructuredSVM::evaluate(const Eigen::VectorXd& x) const
{
	double score = 0.0;  
	for (int i=0; i<svs.size(); ++i)
	{
		const SupportVector& sv = *svs[i];      
    score += sv.b * compute_kernel_score(x, sv.x->x[sv.y]);
	}
  
	return score;
}

double StructuredSVM::test(const Eigen::VectorXd &feature)
{
  return W.dot(feature);
}

double StructuredSVM::validation_test(const Eigen::VectorXd &feature)
{
	int num_psv = 0;
  double validation_score = 0.0;  
	for (int i=0; i<svs.size(); ++i)
	{
		const SupportVector& sv = *svs[i];    
    if (sv.b > 0)
    {
      validation_score += compute_kernel_score(feature, sv.x->x[sv.y]);
      ++num_psv;
    }    
	}  
  validation_score /= (double)num_psv;
	return validation_score;
}

void StructuredSVM::train(const std::vector<Rect> &samples, const std::vector<Eigen::VectorXd> &features, int y)
{
	SupportPattern* sp = new SupportPattern;

	Rect center = samples[y];
	for (int i=0; i<samples.size(); ++i)
	{
		Rect r = samples[i];
		r.x -= center.x;
    r.y -= center.y;
		sp->yv.push_back(r);
	}

	sp->x.resize(samples.size());
  sp->x = features;
	sp->y = y;
	sp->refCount = 0;
	sps.push_back(sp);
	std::cout << (sps.size() - 1) << std::endl;
	ProcessNew(sps.size()-1);
	BudgetMaintenance();	
	for (int i=0; i<10; ++i)
	{
		Reprocess();
		BudgetMaintenance();
	}

  /// Update decision boundary
  W = Eigen::VectorXd(features[0].size());
  W.setZero();

  for (int i=0; i<svs.size(); ++i)
	{
		const SupportVector& sv = *svs[i];        
    W = W + (sv.b * sv.x->x[sv.y]);
	}
}

double StructuredSVM::ComputeDual() const
{
	double d = 0.0;
	for (int i=0; i<svs.size(); ++i)
	{
		const SupportVector* sv = svs[i];
		d -= sv->b*Loss(sv->x->yv[sv->y], sv->x->yv[sv->x->y]);
		for (int j = 0; j < svs.size(); ++j)
      d -= 0.5*sv->b*svs[j]->b*K(i,j);
	}
	return d;
}

void StructuredSVM::SMOStep(int ipos, int ineg)
{
	if (ipos == ineg) return;

	SupportVector* svp = svs[ipos];
	SupportVector* svn = svs[ineg];
	assert(svp->x == svn->x);
	SupportPattern* sp = svp->x;

	if ((svp->g - svn->g) >= 1e-5) 
	{
		double kii = K(ipos, ipos) + K(ineg, ineg) - 2*K(ipos, ineg);
		double lu = (svp->g-svn->g)/kii;
		double l = std::min(lu, C*(int)(svp->y == sp->y) - svp->b);

		svp->b += l;
		svn->b -= l;

		for (int i=0; i<svs.size(); ++i)
		{
			SupportVector* svi = svs[i];
			svi->g -= l*(K(i, ipos) - K(i, ineg));
		}
	}
	
  // update supoort vector accroding to whether b+ and b- are zero
	if (fabs(svp->b) < 1e-8)
	{
		RemoveSupportVector(ipos);
		if (ineg == svs.size())
			ineg = ipos;
	}

	if (fabs(svn->b) < 1e-8)
		RemoveSupportVector(ineg);
}

std::pair<int, double> StructuredSVM::MinGradient(int ind)
{
	const SupportPattern* sp = sps[ind];
	std::pair<int, double> minGrad(-1, DBL_MAX);

  for (int i = 0; i < (int)sp->yv.size(); ++i)
	{
    double grad = -Loss(sp->yv[i], sp->yv[sp->y]) - evaluate(sp->x[i]);
		if (grad < minGrad.second)
		{
			minGrad.first = i;
			minGrad.second = grad;
		}
	}
	return minGrad;
}


void StructuredSVM::ProcessNew(int ind)
{
	int ip = AddSupportVector(sps[ind], sps[ind]->y, -evaluate(sps[ind]->x[sps[ind]->y]));

	std::pair<int, double> minGrad = MinGradient(ind);
	int in = AddSupportVector(sps[ind], minGrad.first, minGrad.second);

	SMOStep(ip, in);
}

void StructuredSVM::ProcessOld()
{
	if (sps.size() == 0) return;
	int ind = rand() % sps.size();

	int ip = -1;
	double maxGrad = -DBL_MAX;
	for (int i=0; i<svs.size(); ++i)
	{
		if (svs[i]->x != sps[ind]) continue;

		const SupportVector* svi = svs[i];
		if (svi->g > maxGrad && svi->b < C*(int)(svi->y == sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
	}
	assert(ip != -1);
	if (ip == -1) return;

	std::pair<int, double> minGrad = MinGradient(ind);
	int in = -1;
	for (int i=0; i<svs.size(); ++i)
	{
		if (svs[i]->x != sps[ind]) continue;

		if (svs[i]->y == minGrad.first)
		{
			in = i;
			break;
		}
	}
	if (in == -1)
		in = AddSupportVector(sps[ind], minGrad.first, minGrad.second);

	SMOStep(ip, in);
}

void StructuredSVM::Reprocess()
{
	ProcessOld();
	for (int i = 0; i < 10; ++i)
		Optimize();
}

void StructuredSVM::Optimize()
{
	if (sps.size() == 0) return;

	int ind = rand() % sps.size();

	int ip = -1;
	int in = -1;
	double maxGrad = -DBL_MAX;
	double minGrad = DBL_MAX;
	for (int i=0; i<svs.size(); ++i)
	{
		if (svs[i]->x != sps[ind]) continue;

		const SupportVector* svi = svs[i];
		if (svi->g > maxGrad && svi->b < C*(int)(svi->y == sps[ind]->y))
		{
			ip = i;
			maxGrad = svi->g;
		}
		if (svi->g < minGrad)
		{
			in = i;
			minGrad = svi->g;
		}
	}
	assert(ip != -1 && in != -1);

	SMOStep(ip, in);
}

int StructuredSVM::AddSupportVector(SupportPattern* x, int y, double g)
{
	SupportVector* sv = new SupportVector;
	sv->b = 0.0;
	sv->x = x;
	sv->y = y;
	sv->g = g;

	std::cout << "sv->x" << std::endl;
	std::cout << sv->x << std::endl;

	std::cout << "sv->y " << std::endl;
	std::cout << sv->y << std::endl;

	std::cout << "sv->g" << std::endl;
	std::cout << sv->g << std::endl;

	int ind = svs.size();
	svs.push_back(sv);
	x->refCount++;

	// update kernel matrix
	for (int i=0; i<ind; ++i)
	{
		K(i,ind) = compute_kernel_score(svs[i]->x->x[svs[i]->y], x->x[y]);
		K(ind,i) = K(i,ind);
	}
	K(ind,ind) = compute_kernel_score(x->x[y]);

	return ind;
}

void StructuredSVM::SwapSupportVectors(int ind1, int ind2)
{
	SupportVector* tmp = svs[ind1];
	svs[ind1] = svs[ind2];
	svs[ind2] = tmp;
	
	Eigen::VectorXd row1 = K.row(ind1);
	K.row(ind1) = K.row(ind2);
	K.row(ind2) = row1;
	
	Eigen::VectorXd col1 = K.col(ind1);
	K.col(ind1) = K.col(ind2);
	K.col(ind2) = col1;
}

void StructuredSVM::RemoveSupportVector(int ind)
{
	svs[ind]->x->refCount--;
	if (svs[ind]->x->refCount == 0)
	{
		// also remove the support pattern
		for (int i=0; i<sps.size(); ++i)
		{
			if (sps[i] == svs[ind]->x)
			{
				delete sps[i];
				sps.erase(sps.begin()+i);
				break;
			}
		}
	}

	if (ind < svs.size()-1)
	{
		SwapSupportVectors(ind, svs.size()-1);
		ind = svs.size()-1;
	}
	delete svs[ind];
	svs.pop_back();
}

void StructuredSVM::BudgetMaintenance()
{
  while (svs.size() > MAX_BUDGET)
		BudgetMaintenanceRemove();	
}

void StructuredSVM::BudgetMaintenanceRemove()
{
	double minVal = DBL_MAX;
	int in = -1;
	int ip = -1;
	for (int i=0; i<svs.size(); ++i)
	{
		if (svs[i]->b < 0.0)
		{
			int j = -1;
			for (int k=0; k<svs.size(); ++k)
			{
				if (svs[k]->b > 0.0 && svs[k]->x == svs[i]->x)
				{
					j = k;
					break;
				}
			}
			double val = svs[i]->b*svs[i]->b*(K(i,i) + K(j,j) - 2.0*K(i,j));
			if (val < minVal)
			{
				minVal = val;
				in = i;
				ip = j;
			}
		}
	}

	svs[ip]->b += svs[in]->b;

	RemoveSupportVector(in);
	if (ip == svs.size())
		ip = in;
	
	if (svs[ip]->b < 1e-8)
		RemoveSupportVector(ip);

	for (int i=0; i<svs.size(); ++i)
	{
		SupportVector& svi = *svs[i];
    svi.g = -Loss(svi.x->yv[svi.y],svi.x->yv[svi.x->y]) - evaluate(svi.x->x[svi.y]);
	}	
}