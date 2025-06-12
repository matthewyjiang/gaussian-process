#ifndef _RBF_KERNEL_H_
#define _RBF_KERNEL_H_

#include "kernel_base.h"

namespace gp {

class RBFKernel : public KernelBase {
private:
  double variance_;    // Signal variance (sigma_f^2)
  double lengthscale_; // Length scale parameter (l)

public:
  RBFKernel(double variance = 1.0, double lengthscale = 1.0);

  Eigen::MatrixXd compute(const Eigen::MatrixXd &X1,
                          const Eigen::MatrixXd &X2) const override;

  std::vector<double> get_params() const override;
  void set_params(const std::vector<double> &params) override;
  size_t num_params() const override;

  std::vector<Eigen::MatrixXd>
  compute_gradients(const Eigen::MatrixXd &X1,
                    const Eigen::MatrixXd &X2) const override;

  // Getters
  double variance() const { return variance_; }
  double lengthscale() const { return lengthscale_; }

  // Setters
  void set_variance(double variance) { variance_ = variance; }
  void set_lengthscale(double lengthscale) { lengthscale_ = lengthscale; }
};

} // namespace gp

#endif // _RBF_KERNEL_H_
