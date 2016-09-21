#include <iostream>
#include <Eigen/Dense>
#include "adkalmanfilter/adkalmanfilter.h"

using namespace std;

template <typename Scalar>
struct adFunctor
{
  static constexpr auto InputSize = 2;

  /*
   * Definitions required by ADKalmanFilter.
   */
  typedef Eigen::Matrix<Scalar, InputSize, 1> InputType;
  typedef Eigen::Matrix<Scalar, InputSize, 1> ValueType;

  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2, typename T3>
  void operator() (const T1 &input, T2 *output, const T3 &controlSignal) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0) + input(1);
    o(1) = input(1) + controlSignal(1);
  }
};

template <typename Scalar>
struct ResFunctor : public ADKalmanFilter::NoOutlierRejection
{
  /*
   * Definitions required by ADKalmanFilter.
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
  typedef Eigen::Matrix<float, 1, 1> RType;

  InputType in;
  RType R;
  in.setZero();
  R.setOnes();

  ADKalmanFilter::ADKalmanFilter< adFunctor<float> > kf;

  kf.innovate< ResFunctor<float> >(in, R);

  return 0;
}
