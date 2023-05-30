#include <iostream>
#include <numeric>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;   // when higher, relys more on measurements -> more quickly adapts to the changes but could be noisy.
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_aug_ = MatrixXd(n_aug_, 2*n_aug_+1);
  x_aug_ = VectorXd(n_aug_);
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  weights_ = VectorXd(2*n_aug_+1);
  lidarNISVec.clear();
  radarNISVec.clear();
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      double px = meas_package.raw_measurements_[0];  
      double py = meas_package.raw_measurements_[1];   

      x_ << px, py, 0, 0, 0;
    }
    else
    {
      // RADAR
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      double px = rho * cos(phi);       
      double py = rho * sin(phi);       // direction of driving x-axis (everything is based on ego coordinate)
      double v = rho_dot; //* cos(phi);    // vx: not the v, but good initialization for highway scenario (better than zero or rho_dot)
      
      x_ << px, py, v, 0, 0;
    }

    weights_.fill(0);
    weights_(0) = lambda_/(lambda_+n_aug_);
    double weight = 0.5/(lambda_+n_aug_);
    for (int i=1; i<2*n_aug_+1; ++i) {  
      weights_(i) = weight;
    }
    // Velocities and angle are uncertain
    P_ <<   1, 0, 0,    0,    0,
            0, 1, 0,    0,    0,
            0, 0, 1000, 0,    0,
            0, 0, 0,    1000, 0,
            0, 0, 0,    0,    1000;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;

  //First: Predict
  //1. Augmented state
  
  x_aug_.head(n_x_) = x_;
  for (int i = n_x_; i < n_aug_; ++i)
  {
    x_aug_(i) = 0;
  }

  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_x_, n_x_) = std_a_*std_a_;
  P_aug_(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  //2. Create Sigma-Points
  // create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug_.col(i+1)        = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }

  Prediction(delta_t);

  //Then: Update (directly use the previous sigma-points)
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    UpdateRadar(meas_package);    
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    
  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i)
  {
    // extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff_(3)> M_PI) x_diff_(3)-=2.*M_PI;
    while (x_diff_(3)<-M_PI) x_diff_(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff_ * x_diff_.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // lidar measures directly: px, py
  int n_z_ = 2;

  // matrix for sigma-points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);

  z_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)  // 2n+1 simga points
  {
    // Measurement model
    Zsig_(0, i) = Xsig_pred_(0, i);  // px
    Zsig_(1, i) = Xsig_pred_(1, i);  // py

    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }
  
  MatrixXd S_ = MatrixXd(n_z_, n_z_);     // Innovation covariance matrix S
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);    // Cross correlation matrix Tc

  // calculate cross correlation matrix Tc and innovation covariance matrix S
  S_.fill(0.0);
  Tc_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  
    // Residual
    VectorXd z_diff_ = Zsig_.col(i) - z_pred_;

    S_ = S_ + weights_(i) * z_diff_ * z_diff_.transpose();

    // State difference
    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff_(3)> M_PI) x_diff_(3)-=2.*M_PI;
    while (x_diff_(3)<-M_PI) x_diff_(3)+=2.*M_PI;

    Tc_ = Tc_ + weights_(i) * x_diff_ * z_diff_.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R_ = MatrixXd(n_z_, n_z_);
  R_ << std_laspx_*std_laspx_, 0,
        0,                     std_laspy_*std_laspy_;
  S_ = S_ + R_;

  // Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();

  // Residual
  VectorXd z_diff_ = meas_package.raw_measurements_ - z_pred_;

  // Update state mean and covariance matrix
  x_ = x_ + K_ * z_diff_;
  P_ = P_ - K_ * S_ * K_.transpose();

  // Calculate Lidar NIS
  double NIS_lidar_threshold = 5.99;
  double NIS_lidar_ = z_diff_.transpose() * S_.inverse() * z_diff_;
  lidarNISVec.push_back(NIS_lidar_);
  double meanNISLidar = std::accumulate(lidarNISVec.begin(), lidarNISVec.end(), 0.0) / lidarNISVec.size();
  if (meanNISLidar >= NIS_lidar_threshold)
  {
    std::cerr << "Warning: NIS_Lidar Passed Threshold of " << NIS_lidar_threshold << ": " << meanNISLidar << std::endl;
  }
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);
  
  // transform sigma points (in object coordinate system) into measurement space (in ego coordinate system)
  // mean predicted measurement
  z_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    if (fabs(p_x) < 0.001 && fabs(p_y) < 0.001)   // handling division by zero
    {
      p_x = 0.001;
      p_y = 0.001;
    }

    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig_(1,i) = atan2(p_y,p_x);                                // phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot

    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }  
  
  MatrixXd S_ = MatrixXd(n_z_, n_z_);     // measurement covariance matrix S
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);     // create matrix for cross correlation Tc

  // calculate cross correlation matrix Tc and innovation covariance matrix S
  S_.fill(0.0);
  Tc_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    
    // residual
    VectorXd z_diff_ = Zsig_.col(i) - z_pred_;
    // angle normalization
    while (z_diff_(1)> M_PI) z_diff_(1)-=2.*M_PI;
    while (z_diff_(1)<-M_PI) z_diff_(1)+=2.*M_PI;
    S_ = S_ + weights_(i) * z_diff_ * z_diff_.transpose();

    // state difference
    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff_(3)> M_PI) x_diff_(3)-=2.*M_PI;
    while (x_diff_(3)<-M_PI) x_diff_(3)+=2.*M_PI;

    Tc_ = Tc_ + weights_(i) * x_diff_ * z_diff_.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R_ = MatrixXd(n_z_,n_z_);
  R_ << std_radr_*std_radr_, 0,                       0,
        0,                   std_radphi_*std_radphi_, 0,
        0,                   0,                       std_radrd_*std_radrd_;
  S_ = S_ + R_;

    // Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();

  // residual
  VectorXd z_diff_ = meas_package.raw_measurements_ - z_pred_;

  // angle normalization
  while (z_diff_(1)> M_PI) z_diff_(1)-=2.*M_PI;
  while (z_diff_(1)<-M_PI) z_diff_(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K_ * z_diff_;
  P_ = P_ - K_*S_*K_.transpose();

  // Calculate Radar NIS
  double NIS_radar_threshold = 7.8;
  double NIS_radar_ = z_diff_.transpose() * S_.inverse() * z_diff_;
  radarNISVec.push_back(NIS_radar_);
  double meanNISRadar = std::accumulate(radarNISVec.begin(), radarNISVec.end(), 0.0) / radarNISVec.size();
  if (meanNISRadar >= NIS_radar_threshold)
  {
    std::cerr << "Warning: NIS_Radar Passed Threshold of " << NIS_radar_threshold << ": " << meanNISRadar << std::endl;
  }
}