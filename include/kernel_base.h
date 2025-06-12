#ifndef _KERNEL_BASE_H_
#define _KERNEL_BASE_H_

#include <Eigen/Dense>
#include <vector>

namespace gp {

class KernelBase {
public:
  virtual ~KernelBase() = default;

  // Compute covariance matrix between two sets of points
  virtual Eigen::MatrixXd compute(const Eigen::MatrixXd &X1,
                                  const Eigen::MatrixXd &X2) const = 0;

  // Compute covariance matrix for single set of points (symmetric)
  virtual Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const {
    return compute(X, X);
  }

  // Get/set hyperparameters
  virtual std::vector<double> get_params() const = 0;
  virtual void set_params(const std::vector<double> &params) = 0;
  virtual size_t num_params() const = 0;

  // Compute gradients w.r.t. hyperparameters
  virtual std::vector<Eigen::MatrixXd>
  compute_gradients(const Eigen::MatrixXd &X1,
                    const Eigen::MatrixXd &X2) const = 0;
};

} // namespace gp

#endif // _KERNEL_BASE_H_
