#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

bool DEBUG = 0;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
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
	std_a_ = 0.75;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.5;

	//DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
	//DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

	/**
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
	 */

	x_<<0,0,0,0,0;
	P_<<1,0,0,0,0,
			0,1,0,0,0,
			0,0,1,0,0,
			0,0,0,1,0,
			0,0,0,0,1;

	n_x_ = 5;
	n_z_ = 3;
	n_aug_=7;
	lambda_ = 3-n_aug_;

	Xsig_pred_ =  MatrixXd(n_x_, 2 * n_aug_ + 1);
	weights_ = VectorXd(2*n_aug_+1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	if(DEBUG)
		cout<<"ProcessMeasurement"<<endl;
	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		/**
		 * Initialize the state x_ with the first measurement.
		 */
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			/**
	      Convert radar from polar to cartesian coordinates and initialize state.
			 */
			float rho = meas_package.raw_measurements_[0];
			float phi = meas_package.raw_measurements_[1];
			float rho_dot = meas_package.raw_measurements_[2];

			float px = rho * cos(phi);
			float py = rho * sin(phi);

			x_<<px,py,0,0,0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			/**
	      Initialize state.
			 */
			float px = meas_package.raw_measurements_[0];
			float py = meas_package.raw_measurements_[1];

			x_<<px,py,0,0,0;
		}

		time_us_ = meas_package.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		if(DEBUG) cout<<"Initialized"<<endl;
		return;
	}

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/
	float dt = (meas_package.timestamp_ - time_us_)/1000000.0;
	time_us_ = meas_package.timestamp_;

	Prediction(dt);

	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	/**
	 * Use the sensor type to perform the update step.
	 * Update the state and covariance matrices.
	 */

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	} else {
		UpdateLidar(meas_package);
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	if(DEBUG)
		cout<<"In Prediction"<<endl;

	/**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	if(DEBUG)
		cout<<"Create augmented mean state"<<endl;
	//create augmented mean state
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_;
	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	MatrixXd bot_right = MatrixXd(2,2);
	bot_right<<std_a_*std_a_, 0,
			0,std_yawdd_*std_yawdd_;
	P_aug.bottomRightCorner(2,2) = bot_right;

	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	A = sqrt(n_aug_+lambda_)*A;

	if(DEBUG)
		cout<<"Create augmented sigma points"<<endl;
	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for(int i=0;i<n_aug_; i++){
		Xsig_aug.col(i+1) = x_aug + A.col(i);
		Xsig_aug.col(n_aug_+i+1) = x_aug - A.col(i);
	}

	if(DEBUG)
		cout<<"Predict sigma points"<<endl;
	//predict sigma points
	for (int i = 0; i< 2*n_aug_+1; i++)
	{
		//extract values for better readability
		double p_x = Xsig_aug(0,i);
		double p_y = Xsig_aug(1,i);
		double v = Xsig_aug(2,i);
		double yaw = Xsig_aug(3,i);
		double yawd = Xsig_aug(4,i);
		double nu_a = Xsig_aug(5,i);
		double nu_yawdd = Xsig_aug(6,i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0,i) = px_p;
		Xsig_pred_(1,i) = py_p;
		Xsig_pred_(2,i) = v_p;
		Xsig_pred_(3,i) = yaw_p;
		Xsig_pred_(4,i) = yawd_p;
	}

	if(DEBUG)
		cout<<"Set weights"<<endl;
	//set weights
	weights_.fill(0.5/(lambda_+n_aug_));
	weights_(0)=lambda_/(lambda_+n_aug_);

	if(DEBUG)
		cout<<"Predict state mean"<<endl;
	//predict state mean
	x_.fill(0.0);
	for(int i=0; i< 2*n_aug_+1; i++){
		x_ = x_ + weights_(i)*Xsig_pred_.col(i);
	}

	if(DEBUG)
		cout<<"Predict state covariance matrix"<<endl;
	//predict state covariance matrix
	P_.fill(0.0);
	for(int i=0; i< 2*n_aug_+1; i++){
		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		if(DEBUG)
			cout<<"Angle normalization:"<<x_diff(3)<<endl;
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
	}
	if(DEBUG)
		cout<<"End of Predict"<<endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
	 */
	if(DEBUG)
		cout<<"UpdateLidar"<<endl;
	VectorXd z = VectorXd(2);
	z<<meas_package.raw_measurements_(0),meas_package.raw_measurements_(1);
	MatrixXd H_ = MatrixXd(2,5);
	H_<<1,0,0,0,0,
			0,1,0,0,0;

	MatrixXd R = MatrixXd(2,2);
	R<< std_laspx_*std_laspx_,0,
			0, std_laspy_*std_laspy_;

	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S_ = H_ * P_ * Ht + R;
	MatrixXd Si = S_.inverse();
	MatrixXd K_ = P_ * Ht * Si;

	x_ = x_ + (K_* y);
	long x_size = x_.size();
	MatrixXd I_ = MatrixXd::Identity(x_size, x_size);
	P_ = (I_ - K_ * H_)*P_;

	long NIS_lidar = y.transpose()*S_.inverse()*y;
	cout<<"NIS Lidar: "<<NIS_lidar<<endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	if(DEBUG)
		cout<<"UpdateRadar"<<endl;
	/**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
	 */
	VectorXd z_ = VectorXd(n_z_);
	z_ << meas_package.raw_measurements_;

	MatrixXd Zsig = MatrixXd(n_z_, 2* n_aug_+1);
	Zsig.fill(0.0);

	if(DEBUG)
		cout<<"Transform sigma points into measurement space"<<endl;
	//transform sigma points into measurement space
	for(int i=0; i<2 * n_aug_ + 1; i++){
		double p_x = Xsig_pred_(0,i);
		double p_y = Xsig_pred_(1,i);
		double v = Xsig_pred_(2,i);
		double si = Xsig_pred_(3,i);

		Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
		Zsig(1,i) = atan2(p_y,p_x);
		Zsig(2,i) = v*(p_x*cos(si)+p_y*sin(si))/Zsig(0,i);
	}

	if(DEBUG)
		cout<<"Calculate mean predicated measurement"<<endl;
	//calculate mean predicted measurement
	VectorXd z_pred = VectorXd(n_z_);
	z_pred.fill(0.0);
	for(int i=0; i<2 * n_aug_ + 1; i++){
		z_pred = z_pred + weights_(i)*Zsig.col(i);
	}

	if(DEBUG)
		cout<<"Calculate innovation covariance matrix S"<<endl;
	//calculate innovation covariance matrix S
	MatrixXd S = MatrixXd(n_z_,n_z_);
	S.fill(0.0);
	for(int i=0; i<2 * n_aug_ + 1; i++){
		MatrixXd diff = Zsig.col(i) - z_pred;
		while(diff(1)>M_PI) diff(1)-= 2*M_PI;
		while(diff(1)<-M_PI) diff(1)+= 2*M_PI;
		S = S + weights_(i)*diff*diff.transpose();
	}

	S(0,0)+=std_radr_*std_radr_;
	S(1,1)+=std_radphi_*std_radphi_;
	S(2,2)+=std_radrd_*std_radrd_;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z_);

	if(DEBUG)
		cout<<"Calculate cross corelation matrix Tc"<<endl;
	//calculate cross correlation matrix
	Tc.fill(0.0);
	for(int i=0; i<2 * n_aug_ + 1; i++){
		MatrixXd X_diff = Xsig_pred_.col(i)-x_;
		if(DEBUG)
			cout<<"Angle normalization of X_diff:"<<X_diff(3)<<endl;
		while (X_diff(3)> M_PI) X_diff(3)-=2.*M_PI;
		while (X_diff(3)<-M_PI) X_diff(3)+=2.*M_PI;

		MatrixXd Z_diff = Zsig.col(i)-z_pred;
		if(DEBUG)
			cout<<"Angle normalization of Z_diff:"<<Z_diff(1)<<endl;
		while (Z_diff(1)> M_PI) Z_diff(1)-=2.*M_PI;
		while (Z_diff(1)<-M_PI) Z_diff(1)+=2.*M_PI;

		Tc = Tc + weights_(i)*X_diff*Z_diff.transpose();
	}

	if(DEBUG)
		cout<<"Calculate Kalman gain K"<<endl;
	//calculate Kalman gain K;
	MatrixXd K = Tc*S.transpose();

	if(DEBUG)
		cout<<"Update state mean and covariance matrix"<<endl;
	//update state mean and covariance matrix
	VectorXd Z_diff = z_-z_pred;
	while (Z_diff(1)> M_PI) Z_diff(1)-=2.*M_PI;
	while (Z_diff(1)<-M_PI) Z_diff(1)+=2.*M_PI;
	x_ = x_ + K*Z_diff;
	P_ = P_ - K*S*K.transpose();

	long NIS_radar = Z_diff.transpose() * S.inverse() * Z_diff;
	cout<<"NIS radar: "<<NIS_radar<<endl;
	if(DEBUG)
		cout<<"State vector"<<x_<<endl;
	if(DEBUG)
		cout<<"Covariance matrix"<<P_<<endl;
}
