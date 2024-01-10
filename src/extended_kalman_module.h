
#ifndef EXTENDED_KALMAN_MODULE_H_
#define EXTENDED_KALMAN_MODULE_H_
#include "extended_kalman_filter.h"
#include "kalman_filter.h"
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "measurement_package.h"

namespace  EKF_MODULE{

typedef int64_t TimeStamp;

// 匀速直线运动，直接使用卡尔曼滤波
// 状态量x,y,vx,vy
// 测量量rho phi rho_dot
class Radar_ptEKF : public ExtendedKalmanFilter {
 public:
  Radar_ptEKF(){};
  const cv::Mat& ProcessMeasurement(const MeasurementPackage &measurement_pack);
  void init(const MeasurementPackage &measurement_pack);
  const cv::Mat& Predict(const cv::Mat& control) override;
  const cv::Mat& Correct(const cv::Mat& measurement,cv::Mat& extra_measurement) override;
  void Transition(const cv::Mat &state_post,
    cv::Mat &state_pre,
    const cv::Mat &control) override;
  void Measurement(const cv::Mat &steate_pre,
                   cv::Mat &measuremet,
                   cv::Mat &extra_measuremet) override;
  void JacobianTrasition(cv::Mat &jacobian_trans,
                         const cv::Mat &control) override;
  void JacobianMeasurement(const cv::Mat &steate_pre,
                           cv::Mat &jacobian_measurement,
                           cv::Mat& extra_measurement) override;

  void JacobianProcessNoise(cv::Mat &jacobian_process_noise,
                            const cv::Mat &control) override;
  void JacobianMeasurementNoise(cv::Mat &jacobian_measurement_noise) override;
  double time_stamp_;
  // cv::Mat error_cov_pre_;
  // cv::Mat error_cov_post_;
  // cv::Mat state_pre_;
  // cv::Mat state_post_;
  bool is_initialized_ = false;
};

}

#endif  // EXTENDED_KALMAN_MODULE_H_