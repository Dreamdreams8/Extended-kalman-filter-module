
#include "kalman_filter.h"
#include <algorithm>

namespace  EKF_MODULE{
void KalmanFilterInterface::Init(int dynam_params, int measure_params,
        int control_params, int type /* = CV_32F */) {
  CV_Assert(dynam_params > 0 && measure_params > 0);
  CV_Assert(type == CV_32F || type == CV_64F);
  control_params = std::max<int>(control_params, 0);

  state_pre_ = cv::Mat::zeros(dynam_params, 1, type);
  state_post_ = cv::Mat::zeros(dynam_params, 1, type);

  transition_matrix_ = cv::Mat::eye(dynam_params, dynam_params, type);
  process_noise_cov_ = cv::Mat::eye(dynam_params, dynam_params, type);

  measurement_matrix_ = cv::Mat::zeros(measure_params, dynam_params, type);
  measurement_noise_cov_ = cv::Mat::eye(measure_params, measure_params, type);

  error_cov_pre_ = cv::Mat::zeros(dynam_params, dynam_params, type);
  error_cov_post_ = cv::Mat::zeros(dynam_params, dynam_params, type);

  gain_ = cv::Mat::zeros(dynam_params, measure_params, type);

  if (control_params > 0)
    control_matrix_ = cv::Mat::zeros(dynam_params, control_params, type);
  else
    control_matrix_.release();
}

//////////////////////////////////////////////////////////////////////////

void KalmanFilter::Init(int dynam_params, int measure_params,
                        int control_params, int type /* = CV_32F */) {
  KalmanFilterInterface::Init(dynam_params, measure_params,
          control_params, type);

  temp1_.create(dynam_params, dynam_params, type);
  temp2_.create(measure_params, dynam_params, type);
  temp3_.create(measure_params, measure_params, type);
  temp4_.create(measure_params, dynam_params, type);
  temp5_.create(measure_params, 1, type);
}

const cv::Mat& KalmanFilter::Predict(const cv::Mat& control) {
  // 更新预测值: x'(k) = A*x(k)
  state_pre_ = transition_matrix_ * state_post_;

  if (!control.empty()){
    // x'(k) = x'(k) + B*u(k)
    state_pre_ += control_matrix_  *  control;
  }

  // temp1_ = A*P(k)
  temp1_ = transition_matrix_  *  error_cov_post_;

  // P'(k) = temp1_*At + Q
  // 输出error_cov_pre_，两个1分别代表+号左右两边的权重
  cv::gemm(temp1_, transition_matrix_, 1.0f,
           process_noise_cov_, 1.0f, error_cov_pre_, cv::GEMM_2_T);

  // handle the case when there will be measurement before the next predict.
  state_pre_.copyTo(state_post_);
  error_cov_pre_.copyTo(error_cov_post_);

  return state_pre_;
}

const cv::Mat& KalmanFilter::Correct(const cv::Mat& measurement,
        cv::Mat& extra_measurement) {
  // temp2_ = H*P'(k)
  temp2_ = measurement_matrix_ * error_cov_pre_;

  // temp3_ = temp2_*Ht + R
  cv::gemm(temp2_, measurement_matrix_, 1.0f,
           measurement_noise_cov_, 1.0f, temp3_, cv::GEMM_2_T);

  // temp4 = inv(temp3)*temp2 = Kt(k)
  // solve相当于求解Ax=B的解
  cv::solve(temp3_, temp2_, temp4_, cv::DECOMP_SVD);
  // K(k)
  gain_ = temp4_.t();

  // temp5_ = z(k) - H*x'(k)
  temp5_ = measurement - measurement_matrix_ * state_pre_;

  // x(k) = x'(k) + K(k)*temp5_
  state_post_ = state_pre_ + gain_*temp5_;

  // P(k) = P'(k) - K(k)*temp2_
  error_cov_post_ = error_cov_pre_ - gain_*temp2_;

  return state_post_;
}
}