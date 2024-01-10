#include <iostream>
#include "tools.h"
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace cv;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  Hj << 0,0,0,0,
        0,0,0,0,
        0,0,0,0;

  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  if(fabs(c1) < 0.0001){
    std::cout << "Function CalculateJacobian() has Error: Division by Zero" << std::endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/c2),                (py/c2),                0,      0,
        -(py/c1),               (px/c1),                0,      0, 
        py*(vx*py - vy*px)/c3,  px*(px*vy - py*vx)/c3,  px/c2,  py/c2;

  return Hj;

}



void Tools::Debug_opencv_show(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth){
    if(estimations.size() != ground_truth.size() || estimations.size() == 0){
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return ;
    }         
    std::cout << "sucess estimation or ground_truth data" << std::endl;
    float offset_xratio = 0.5,img_multiple = 10,v_multiple = 30;
    int rows = 600;
    int cols = 600;
    int point_size = 3;
    int rows_add = rows * offset_xratio;
    cv::Mat img1= cv::Mat::zeros(Size(cols, rows), CV_8UC3);
    img1.setTo(255);         
    cv::Mat img2 = cv::Mat::zeros(Size(cols, rows), CV_8UC3);
    img2.setTo(255);          
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        // std::cout << "x: " << estimations[i][0]  <<  "        y:  "<< estimations[i][1]   << std::endl;
        // std::cout << "gx: " << ground_truth[i][0]  <<  "        gy:  "<< ground_truth[i][1] << std::endl;
        // std::cout << "vx: " << estimations[i][2]  <<  "        vy:  "<< estimations[i][3]   << std::endl;
        // std::cout << "gvx: " << ground_truth[i][2]  <<  "        gvy:  "<< ground_truth[i][3] << std::endl;
        cv::Point p1(estimations[i][0] * img_multiple + cols / 2.0,estimations[i][1] * img_multiple + rows_add);
        cv::Point p2(ground_truth[i][0] * img_multiple + cols / 2.0,ground_truth[i][1]* img_multiple + rows_add);
        cv::Point p3(estimations[i][2] * v_multiple + cols / 2.0,estimations[i][3] * v_multiple + rows_add);
        cv::Point p4(ground_truth[i][2] * v_multiple + cols / 2.0,ground_truth[i][3] * v_multiple + rows_add);
        cv::circle(img1, p1, point_size, Scalar(100, 100, 100), -1);
        cv::circle(img1, p2, point_size, Scalar(100, 255, 0), -1);
        cv::circle(img2, p3, point_size, Scalar(100, 100, 100), -1);
        cv::circle(img2, p4, point_size, Scalar(100, 255, 0), -1);     

    }                        
    cv::imshow("位置", img1);   // 原始图形
    cv::imshow("速度", img2);   // 原始图形
    cvWaitKey();     
    return;        
    }