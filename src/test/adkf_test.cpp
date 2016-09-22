#include <iostream>
#include <Eigen/Dense>
#include "adkalmanfilter/adkalmanfilter.h"

using namespace std;

template <typename Scalar>
struct predFunctor
{
  static constexpr auto InputSize = 2;

  /*
   * Definitions required for input and output type.
   * Used by the AutoDiff to find Jacobians.
   */
  typedef Eigen::Matrix<Scalar, InputSize, 1> InputType;
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
struct ResFunctor : public ADKalmanFilter::NoOutlierRejection
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

  InputType in = InputType::Zero();
  MeasType meas = MeasType::Zero();
  RType R = RType::Identity();
  QType Q = QType::Identity();
  HType H = HType::Ones();
  in.setZero();

  ADKalmanFilter::ADKalmanFilter< predFunctor<float> > kf(in, Q);

  kf.predict(Q, &in);

  kf.update< ResFunctor<float> >(meas, R, &H);
  kf.update< ResFunctor<float> >(meas, R);

  return 0;
}
