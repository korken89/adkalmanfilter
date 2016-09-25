#include <iostream>
#include <Eigen/Dense>
#include "adkalmanfilter/adkalmanfilter.h"

using namespace std;


template <typename Scalar>
struct predFunctor : public ADKalmanFilter::baseFunctor<Scalar, 2>
{
  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2, typename T3>
  void operator() (const T1 &input, T2 *output, const T3 &controlSignal) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0) + 1e-2*input(1);
    o(1) = controlSignal(0);
  }
};

template <typename Scalar>
struct measFunctor : public ADKalmanFilter::baseFunctor<Scalar, 2, 1>,
                     public ADKalmanFilter::MahalanobisOutlierRejection<10>
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
  typedef Eigen::Matrix<float, 2, 1> InputType;
  typedef Eigen::Matrix<float, 1, 1> MeasType;
  typedef Eigen::Matrix<float, 1, 1> RType;
  typedef Eigen::Matrix<float, 2, 2> QType;
  typedef Eigen::Matrix<float, 1, 2> HType;
  typedef Eigen::Matrix<float, 2, 2> FType;

  std::srand((unsigned int) time(0));

  InputType in = InputType::Zero();
  InputType u = InputType::Zero();
  MeasType meas;
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

  kf.predict(F, u, Q);
  //kf.predict(F, Q);
  kf.predict(u, Q);
  //kf.predict(Q);

  kf.getState(out);
  kf.getStateCovariance(P);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;
  std::cout << "u = " << std::endl << u << std::endl << std::endl;
  std::cout << "P = " << std::endl << P << std::endl << std::endl;

  std::cout << "Running a lot of iterations... " << std::endl << std::endl;
  for (auto i = 0; i < 100; i++)
  {
    meas = MeasType::Random();
    kf.predict(u, Q);
    kf.update< measFunctor<float> >(meas, R);
  }

  kf.getState(out);
  kf.getStateCovariance(P);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;
  std::cout << "P = " << std::endl << P << std::endl << std::endl;

  return 0;
}
