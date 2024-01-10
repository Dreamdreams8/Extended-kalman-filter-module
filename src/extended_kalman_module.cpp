#include <iostream>
#include "extended_kalman_module.h"

namespace  EKF_MODULE{
const cv::Mat& Radar_ptEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

    if(!is_initialized_){
      init(measurement_pack);
    }

    cv::Mat control = cv::Mat::zeros(3, 1, CV_32F);
    control.at<float>(0) = 0;  // speed
    control.at<float>(1) =0;  // yaw_rate 
    control.at<float>(2) = (measurement_pack.timestamp_ - this->time_stamp_) / 1000000.0; // dt
    this->time_stamp_ = measurement_pack.timestamp_; 
    cv::Mat state_pre = Predict(control);
    cv::Mat measurement = cv::Mat::zeros(3, 1, CV_32F);//3
    cv::Mat extra_measurement = cv::Mat::zeros(3, 1, CV_32F);
   if(measurement_pack.sensor_type_ == MeasurementPackage::RADAR){
        float rho = measurement_pack.raw_measurements_[0];      // range: radial distance from origin
        float phi = measurement_pack.raw_measurements_[1];      // bearing: angle between rho and x axis
        float rho_dot = measurement_pack.raw_measurements_[2]; 
        measurement.at<float>(0) = rho;  
        measurement.at<float>(1) = phi; 
        measurement.at<float>(2) =  rho_dot;
   }                                                  
  const cv::Mat &mcorrect  = Correct(measurement,extra_measurement);
  return mcorrect;

}

void Radar_ptEKF::init(const MeasurementPackage &measurement_pack){
  // x,y,vx,vy
  int dynam_params = 4;
  // x, y
  int measure_params = 3;
  // speed,yaw_rate,dt
  int control_params = 3;

  int type = CV_32FC1;

  this->time_stamp_ = measurement_pack.timestamp_;  // 此处要外部传入
  
  ExtendedKalmanFilter::Init(dynam_params, measure_params,
    control_params, type);    

  // control_matrix_ will not use
  control_matrix_.release();
   
  // init process_noise_cov_
  cv::setIdentity(process_noise_cov_, cv::Scalar::all(0.2f));
  cv::setIdentity(measurement_noise_cov_, cv::Scalar::all(0.009f));

   // 初始状态赋值
   if(measurement_pack.sensor_type_ == MeasurementPackage::RADAR){
      float rho = measurement_pack.raw_measurements_[0];      // range: radial distance from origin
      float phi = measurement_pack.raw_measurements_[1];      // bearing: angle between rho and x axis
      float rho_dot = measurement_pack.raw_measurements_[2];  // radial velocity: change of rho

      state_post_.at<float>(0, 0) = rho * cos(phi);   // x
      state_post_.at<float>(1, 0) = rho * sin(phi);   // y
      state_post_.at<float>(2, 0) = 0.0f;   // vx
      state_post_.at<float>(3, 0) = 0.0f;   // vy
   }


  state_pre_ = state_post_.clone();

  error_cov_post_.setTo(cv::Scalar::all(0.0f));
  error_cov_post_.at<float>(0, 0) = 0.1f;  // x
  error_cov_post_.at<float>(1, 1) = 0.1f;  // y
  error_cov_post_.at<float>(2, 2) = 1000.0f;  // vx
  error_cov_post_.at<float>(3, 3) = 1000.0f;   // vy

  // error_cov_post_ = error_cov_post_ * error_cov_post_;
  error_cov_pre_ = error_cov_post_.clone();
  is_initialized_ = true;
}

const cv::Mat& Radar_ptEKF::Predict(const cv::Mat& control){
  return ExtendedKalmanFilter::Predict(control);
}


void Radar_ptEKF::Transition(const cv::Mat & state_post, cv::Mat & state_pre,
                         const cv::Mat & control) {
  cv::setIdentity(transition_matrix_);
  JacobianTrasition(transition_matrix_, control);
  state_pre = transition_matrix_ * state_post;
}

void Radar_ptEKF::JacobianTrasition(cv::Mat & jacobian_trans,
                                const cv::Mat & control) {
  float speed = control.at<float>(0);
  float yaw_rate = control.at<float>(1);
  float dt = control.at<float>(2);

  // px
  jacobian_trans.at<float>(0, 0) = 1.0f; 
  jacobian_trans.at<float>(0, 1) = 0;
  jacobian_trans.at<float>(0,  2) = dt;  
  jacobian_trans.at<float>(0,  3) = 0.0f;

  // py
  jacobian_trans.at<float>(1,  0) = 0.0f;
  jacobian_trans.at<float>(1,  1) = 1.0f;
  jacobian_trans.at<float>(1,  2) = 0.0f;
  jacobian_trans.at<float>(1,   3) = dt;

  // vx
  jacobian_trans.at<float>(2,   0) = 0.0f;
  jacobian_trans.at<float>(2,   1) = 0.0f;
  jacobian_trans.at<float>(2,    2) = 1.0f; 
  jacobian_trans.at<float>(2,    3) = 0.0f;

  // vy
  jacobian_trans.at<float>(3,    0) = 0.0f;
  jacobian_trans.at<float>(3,    1) = 0.0f;
  jacobian_trans.at<float>(3,    2) = 0.0f;
  jacobian_trans.at<float>(3,    3) = 1.0f;

}


void Radar_ptEKF::Measurement(const cv::Mat & state_pre,
                          cv::Mat & measuremet, cv::Mat& extra_measurement) {
  float px, py, vx, vy;
  px = state_pre.at<float>(0,0);
  py = state_pre.at<float>(1,0);
  vx = state_pre.at<float>(2,0);
  vy = state_pre.at<float>(3,0);

  measuremet.at<float>(0) = sqrt(px * px+py * py);
  measuremet.at<float>(1) = atan2(py,px);   // returns values between -pi and pi
  measuremet.at<float>(2) = (px*vx + py * vy)/sqrt(px*px+py*py);

}

void Radar_ptEKF::JacobianMeasurement(const cv::Mat & state_pre,
                                  cv::Mat & jacobian_measurement,
                                  cv::Mat& extra_measurement) {
  jacobian_measurement.setTo(cv::Scalar(0)); 
  float px, py, vx, vy;
  px = state_pre.at<float>(0,0);
  py = state_pre.at<float>(1,0);
  vx = state_pre.at<float>(2,0);
  vy = state_pre.at<float>(3,0);

  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);
  if(fabs(c1) < 0.0001){
    return;
  }

  jacobian_measurement.at<float>(0,0) =  px/c2;
  jacobian_measurement.at<float>(0,1) = py/c2;
  jacobian_measurement.at<float>(1,0) = -py/c1;
  jacobian_measurement.at<float>(1,1) = px/c1;
  jacobian_measurement.at<float>(2,0) = py*(vx*py - vy*px)/c3;
  jacobian_measurement.at<float>(2,1) = px*(px*vy - py*vx)/c3;
  jacobian_measurement.at<float>(2,2) = px/c2;
  jacobian_measurement.at<float>(2,3) = py/c2;
}

const cv::Mat& Radar_ptEKF::Correct(const cv::Mat& measurement,
                                cv::Mat& extra_measurement) {                           
  const cv::Mat &mcorrect = ExtendedKalmanFilter::Correct(measurement,
                                                          extra_measurement);

  // for literately correct
  error_cov_post_.copyTo(error_cov_pre_);
  state_post_.copyTo(state_pre_);

  return mcorrect;
}

// 过程噪声可以设置跟控制变量相关
void Radar_ptEKF::JacobianProcessNoise(cv::Mat & jacobian_process_noise,
                                   const cv::Mat & control) {
  float speed = control.at<float>(0);
  float yaw_rate = fabs(control.at<float>(1));
  float dt = control.at<float>(2);

  // 此处可根据实际效果调整影响因子
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  float noise_ax = 9.0;
  float noise_ay = 9.0;  

  // process_noise_cov_.at<float>(0, 0) = dt_4/4*noise_ax; 
  // process_noise_cov_.at<float>(0, 1) = 0;
  // process_noise_cov_.at<float>(0,  2) = dt_3/2*noise_ax;  
  // process_noise_cov_.at<float>(0,  3) = 0.0f;

  // process_noise_cov_.at<float>(1,  0) = 0.0f;
  // process_noise_cov_.at<float>(1,  1) = dt_4/4*noise_ay;
  // process_noise_cov_.at<float>(1,  2) = 0.0f;
  // process_noise_cov_.at<float>(1,  3) = dt_3/2*noise_ay;

  // process_noise_cov_.at<float>(2,   0) = dt_3/2*noise_ax;
  // process_noise_cov_.at<float>(2,   1) = 0.0f;
  // process_noise_cov_.at<float>(2,    2) = dt_2*noise_ax; 
  // process_noise_cov_.at<float>(2,    3) = 0.0f;

  // process_noise_cov_.at<float>(3,    0) = 0.0f;
  // process_noise_cov_.at<float>(3,    1) = dt_3/2*noise_ay;
  // process_noise_cov_.at<float>(3,    2) = 0.0f;
  // process_noise_cov_.at<float>(3,    3) = dt_2*noise_ay;


  cv::setIdentity(jacobian_process_noise, cv::Scalar::all(1.0f));
  // jacobian_process_noise.at<float>(0, 0) = 1.0f;
  // jacobian_process_noise.at<float>(1, 1) = 1.0f;
  // jacobian_process_noise.at<float>(2, 2) = 1.0f;
  // jacobian_process_noise.at<float>(3, 3) = 1.0f;

}

void Radar_ptEKF::JacobianMeasurementNoise(cv::Mat & jacobian_measurement_noise) {
  cv::setIdentity(jacobian_measurement_noise, 1.0f);
  measurement_noise_cov_ .at<float>(2, 2) = 0.0009f;
}

}
