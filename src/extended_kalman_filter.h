#ifndef EXTENDED_KALMAN_FILTER_H_
#define EXTENDED_KALMAN_FILTER_H_
#include "kalman_filter.h"
#include <vector>
#include <opencv2/opencv.hpp>

namespace  EKF_MODULE{
class ExtendedKalmanFilter : public KalmanFilterInterface {
 public:
  ExtendedKalmanFilter() {
  }

  virtual void Init(int dynam_params, int measure_params,
         int control_params, int type = CV_32F);

  virtual const cv::Mat& Predict(const cv::Mat& control);
  virtual const cv::Mat& Correct(const cv::Mat& measurement,
         cv::Mat& extra_measurement);

  virtual void Transition(const cv::Mat &state_post,
                          cv::Mat &state_pre,
                          const cv::Mat &control) = 0;
  virtual void Measurement(const cv::Mat &steate_pre,
                           cv::Mat &measuremet,
                           cv::Mat &extra_measuremet) = 0;
  virtual void JacobianTrasition(cv::Mat &jacobian_trans,
                                 const cv::Mat &control) = 0;
  virtual void JacobianMeasurement(const cv::Mat &steate_pre,
                                   cv::Mat &jacobian_measurement,
                                   cv::Mat& extra_measurement) = 0;
  virtual void JacobianProcessNoise(cv::Mat &jacobian_process_noise,
                                    const cv::Mat &control) = 0;
  virtual void JacobianMeasurementNoise(cv::Mat &jacobian_measurement_noise) = 0;

 protected:
  cv::Mat jacobian_trasition_;
  cv::Mat jacobian_measurement_;
  cv::Mat jacobian_process_noise_;
  cv::Mat jacobian_measurement_noise_;

  cv::Mat process_noise_tmp_;
  cv::Mat measurement_noise_tmp_;

  // temporary matrices
  cv::Mat temp_state_;
  cv::Mat temp_measurement_;
  cv::Mat temp1_;
  cv::Mat temp2_;
  cv::Mat temp3_;
  cv::Mat temp4_;
  cv::Mat temp5_;
};
}
#endif  // EXTENDED_KALMAN_FILTER_H_
