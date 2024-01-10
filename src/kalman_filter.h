
#ifndef COMON_KALMAN_FILTER_H_
#define COMON_KALMAN_FILTER_H_

#include <opencv2/opencv.hpp>

#ifdef LOW_GCC_VERSION
#define override
#endif


namespace  EKF_MODULE{

class KalmanFilterInterface {
 public:
  KalmanFilterInterface() {
  }
  virtual ~KalmanFilterInterface() {
  }

  virtual void Init(int dynam_params, int measure_params,
                    int control_params, int type = CV_32F);

  virtual const cv::Mat& Predict(const cv::Mat& control) = 0;
  virtual const cv::Mat& Correct(const cv::Mat& measurement,
                                 cv::Mat& extra_measurement) = 0;


  cv::Mat state_pre_;                                   // 预测值(x'(k)): x(k)=A*x(k-1)+B*u(k)
  cv::Mat state_post_;                                 // 状态值 (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
  cv::Mat transition_matrix_;                  // 状态转移矩阵 (A)
  cv::Mat control_matrix_;                        // 控制矩阵 (B)
  cv::Mat measurement_matrix_;          // 测量矩阵 (H)
  cv::Mat process_noise_cov_;                // 系统误差 (Q)
  cv::Mat measurement_noise_cov_;   // 测量误差 (R)
  cv::Mat error_cov_pre_;                           // 最小均方误差 (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
  cv::Mat gain_;                                                // 卡尔曼增益 (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
  cv::Mat error_cov_post_;                         // 修正的最小均方误差(P(k)): P(k)=(I-K(k)*H)*P'(k)
};

class KalmanFilter : public KalmanFilterInterface {
 public:
  KalmanFilter() {
  }

  void Init(int dynam_params, int measure_params,
            int control_params, int type = CV_32F) override;

  const cv::Mat& Predict(const cv::Mat& control) override;
  const cv::Mat& Correct(const cv::Mat& measurement,
                         cv::Mat& extra_measurement) override;

 public:
  // temporary matrices
  cv::Mat temp1_;
  cv::Mat temp2_;
  cv::Mat temp3_;
  cv::Mat temp4_;
  cv::Mat temp5_;
};   // KalmanFilter


}
#endif  // COMON_KALMAN_FILTER_H_
