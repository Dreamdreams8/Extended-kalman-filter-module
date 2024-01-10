#include "extended_kalman_filter.h"
#include <vector>
#include <algorithm>

namespace  EKF_MODULE{

void ExtendedKalmanFilter::Init(int dynam_params, int measure_params,
                                int control_params, int type/* = CV_32F*/) {
  KalmanFilterInterface::Init(dynam_params, measure_params,
                              control_params, type);

  jacobian_trasition_.create(dynam_params, dynam_params, type);
  jacobian_measurement_.create(measure_params, dynam_params, type);

  temp_state_.create(dynam_params, 1, type);
  temp_measurement_.create(measure_params, 1, type);

  jacobian_process_noise_.create(dynam_params, dynam_params, type);
  jacobian_measurement_noise_.create(measure_params, measure_params,  type);

  process_noise_tmp_.create(dynam_params, dynam_params, type);
  measurement_noise_tmp_.create(measure_params, measure_params, type);

  temp1_.create(dynam_params, dynam_params, type);
  temp2_.create(measure_params, dynam_params, type);
  temp3_.create(measure_params, measure_params, type);
  temp4_.create(measure_params, dynam_params, type);
  temp5_.create(measure_params, 1, type);
}

const cv::Mat& ExtendedKalmanFilter::Predict(const cv::Mat& control) {
  // P'(k) = JF*P(k-1)*JFt + Q(k-1)
  JacobianTrasition(jacobian_trasition_, control);

  // update error covariance matrices: temp1 = JF*P(k)
  temp1_ = jacobian_trasition_* error_cov_post_;

  // Q = W(k) * Q(k-1) * Wt(k)
  JacobianProcessNoise(jacobian_process_noise_, control);
  cv::gemm(jacobian_process_noise_, process_noise_cov_, 1.0f,
           cv::Mat(), 1.0f, process_noise_tmp_);
  cv::gemm(process_noise_tmp_, jacobian_process_noise_, 1.0f,
    cv::Mat(), 1.0f, process_noise_tmp_, cv::GEMM_2_T);

  // P'(k) = temp1*JFt + Q
  cv::gemm(temp1_, jacobian_trasition_, 1.0f,
           process_noise_tmp_, 1.0f, error_cov_pre_, cv::GEMM_2_T);

  // calculate x'(k) = f( x(k-1), u(k-1) )
  Transition(state_post_, state_pre_, control);
  // x'(k) = x'(k) + B*u(k)
  if (!control.empty() && !control_matrix_.empty())
    state_pre_ += control_matrix_ * control;

  // avoid no correct
  state_pre_.copyTo(state_post_);
  // error_cov_pre_.copyTo(error_cov_post_);
  return state_pre_;
}

const cv::Mat& ExtendedKalmanFilter::Correct(const cv::Mat& measurement,
  cv::Mat& extra_measurement) {
  // K(k) = P'(k) * JHt * ( JH * P'(k) * JHt + R(k) )^–1
  // temp2 = JH * P'(k)
  JacobianMeasurement(state_pre_, jacobian_measurement_, extra_measurement);
  temp2_ = jacobian_measurement_* error_cov_pre_;
  // temp3 = temp2*Ht
  cv::gemm(temp2_, jacobian_measurement_, 1.0f,
    cv::Mat(), 1.0f, temp3_, cv::GEMM_2_T);
  // R(k) = V(k) * R * Vt(k)
  JacobianMeasurementNoise(jacobian_measurement_noise_);
  cv::gemm(jacobian_measurement_noise_, measurement_noise_cov_, 1.0f,
    cv::Mat(), 1.0f, measurement_noise_tmp_);
  // temp3 = temp3 + R, <-- JH*P'(k)*JHt + R(k)
  cv::gemm(measurement_noise_tmp_, jacobian_measurement_noise_, 1.0f,
    temp3_, 1.0f, temp3_, cv::GEMM_2_T);

  cv::gemm(jacobian_measurement_, error_cov_pre_, 1.0f,
        cv::Mat(), 0.0f, temp5_, cv::GEMM_2_T);

  cv::solve(temp3_, temp5_, temp4_, cv::DECOMP_SVD);
  // K(k)
  gain_ = temp4_.t();

  // 2. x(k) = x'(k) + K(k) * ( z(k) – h(x'(k), 0) )
  // temp5 = z(k) - h(x'(k))
  Measurement(state_pre_, temp_measurement_, extra_measurement);
  temp5_ = measurement - temp_measurement_;

  // 其他场景根据实际需要删除
  // normalize the angle between -pi to pi
  while(temp5_.at<float>(1,0) > M_PI){
    temp5_.at<float>(1,0)  -= 2 * M_PI;
  }

  while(temp5_.at<float>(1,0)  < -M_PI){
    temp5_.at<float>(1,0) += 2 * M_PI;
  }

  // x(k) = x'(k) + K(k) * temp5
  state_post_ = state_pre_ + gain_ * temp5_;
  // 3. P(k) = P'(k) - K(k) * JH * P'(k)
  error_cov_post_ = error_cov_pre_ - gain_ * temp2_;
  return state_post_;
}

}
