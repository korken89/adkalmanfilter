#include <iostream>
#include <Eigen/Dense>
#include "adkalmanfilter/adkalmanfilter.h"

using namespace std;

template <typename Scalar>
struct predFunctor
{
  /*
   * Definitions required for input and output type.
   * Used by the AutoDiff to find Jacobians.
   */
  typedef Eigen::Matrix<Scalar, 2, 1> InputType;
  typedef Eigen::Matrix<Scalar, InputType::RowsAtCompileTime, 1> ValueType;

  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2, typename T3>
  void operator() (const T1 &input, T2 *output, const T3 &controlSignal) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0) + input(1);
    o(1) = input(1) + controlSignal(0);
  }
};

template <typename Scalar>
struct measFunctor : public ADKalmanFilter::MahalanobisOutlierRejection<10>
{
  /*
   * Definitions required for input and output type.
   * Used by the AutoDiff to find Jacobians.
   */
  typedef Eigen::Matrix<Scalar, 2, 1> InputType;
  typedef Eigen::Matrix<Scalar, 1, 1> ValueType;

  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2>
  void operator() (const T1 &input, T2 *output) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0) + input(1);
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
  HType H = HType::Ones();
  FType F = FType::Ones();
  F(1,0) = 0;
  in.setZero();

  ADKalmanFilter::ADKalmanFilter< predFunctor<float> > kf(in, Q);

  typedef typename ADKalmanFilter::ADKalmanFilter< predFunctor<float> >::StateType StateType;

  StateType out;

  kf.predict(Q, F, u);
  //kf.predict(Q, F);
  kf.predict(Q, u);
  //kf.predict(Q);

  meas = MeasType::Random();
  if(kf.update< measFunctor<float> >(meas, R, H))
    std::cout << "Measurement accepted!" << std::endl;
  else
    std::cout << "Measurement rejected!" << std::endl;

  kf.getState(out);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;

  meas = MeasType::Random()*100;
  if(kf.update< measFunctor<float> >(meas, R, H))
    std::cout << "Measurement accepted!" << std::endl;
  else
    std::cout << "Measurement rejected!" << std::endl;

  kf.getState(out);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;

  meas = MeasType::Random();
  if(kf.update< measFunctor<float> >(meas, R))
    std::cout << "Measurement accepted!" << std::endl;
  else
    std::cout << "Measurement rejected!" << std::endl;

  kf.getState(out);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;

  meas = MeasType::Random()*100;
  if(kf.update< measFunctor<float> >(meas, R))
    std::cout << "Measurement accepted!" << std::endl;
  else
    std::cout << "Measurement rejected!" << std::endl;

  kf.getState(out);
  std::cout << "x = " << std::endl << out << std::endl << std::endl;

  return 0;
}
