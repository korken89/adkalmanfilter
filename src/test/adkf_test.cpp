#include <iostream>
#include <Eigen/Dense>
#include "adkalmanfilter/adkalmanfilter.h"

using namespace std;

template <typename Scalar>
struct predFunctor : public ADKalmanFilter::BaseFunctor<Scalar, 2>
{
  typedef Eigen::Matrix<Scalar, 1, 1> ControlType;

  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2>
  void operator() (const T1 &input, T2 *output,
                   const Eigen::Ref<const ControlType> &controlSignal,
                   const Scalar &dt) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0) + dt*input(1);
    o(1) = controlSignal(0);
  }
};

template <typename Scalar>
struct measFunctor : public ADKalmanFilter::BaseFunctor<Scalar, 2, 1>,
                     public ADKalmanFilter::MahalanobisOutlierRejection<50>
{
  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2>
  void operator() (const T1 &input, T2 *output) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0);
  }
};

int main(int argc, char *argv[])
{
  typedef Eigen::Matrix<float, 2, 1> StateType;
  typedef Eigen::Matrix<float, 1, 1> ContType;
  typedef Eigen::Matrix<float, 1, 1> MeasType;
  typedef Eigen::Matrix<float, 1, 1> RType;
  typedef Eigen::Matrix<float, 2, 2> QType;
  typedef Eigen::Matrix<float, 1, 2> HType;
  typedef Eigen::Matrix<float, 2, 2> FType;

  std::srand((unsigned int) time(0));

  double dt = 1e-2;

  StateType in = StateType::Zero();
  ContType u = ContType::Zero();
  MeasType meas = MeasType::Random();
  RType R = RType::Identity();
  QType Q = QType::Identity();
  HType H = HType::Identity();
  FType F = FType::Ones();
  F(1,0) = 0;
  in.setZero();

  ADKalmanFilter::ADKalmanFilter< predFunctor<float> > kf(in, Q);

  typedef typename ADKalmanFilter::ADKalmanFilter< predFunctor<float> >::StateType StateType;
  typedef typename ADKalmanFilter::ADKalmanFilter< predFunctor<float> >::StateCovarianceType StateCovarianceType;

  StateType out;
  StateCovarianceType P;

  kf.predict(F, Q, u, dt);
  //kf.predict(F, Q);
  kf.predictAD(Q, u, dt);
  //kf.predict(Q);
  kf.update< measFunctor<float> >(H, meas, R);

  kf.getState(out);
  kf.getStateCovariance(P);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;
  std::cout << "u = " << std::endl << u << std::endl << std::endl;
  std::cout << "P = " << std::endl << P << std::endl << std::endl;

  std::cout << "Running a lot of iterations... " << std::endl << std::endl;
  for (auto i = 0; i < 1e5; i++)
  {
    meas = MeasType::Random();
    kf.predictAD(Q, u, dt);
    kf.updateAD< measFunctor<float> >(meas, R);
  }

  kf.getState(out);
  kf.getStateCovariance(P);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;
  std::cout << "P = " << std::endl << P << std::endl << std::endl;

  return 0;
}
