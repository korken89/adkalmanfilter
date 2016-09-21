#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#ifndef _AD_KF
#define _AD_KF

namespace ADKalmanFilter {

/*
 * Used for defining no outlier rejection.
 */
struct NoOutlierRejection
{
  template <typename... Params>
  bool rejectMeasurement(const Params &...) const
  {
    return false;
  }
};

template <typename PredictionFunctor>
class ADKalmanFilter : public PredictionFunctor
{

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*
   * All functors have the following defined types:
   *
   * Functor::InputType
   * Functor::ValueType
   *
   * We use this to infere state size at compile time.
   *
   */

  typedef typename PredictionFunctor::InputType::Scalar Scalar;
  typedef typename PredictionFunctor::InputType StateType;
  typedef Eigen::Matrix<Scalar, StateType::RowsAtCompileTime,
                        StateType::RowsAtCompileTime> CovarianceType;
  typedef Eigen::Matrix<Scalar, StateType::RowsAtCompileTime,
                        StateType::RowsAtCompileTime> PredictionJacobianType;

  ADKalmanFilter() { }

  ADKalmanFilter(const Eigen::MatrixBase<StateType> &x_init,
                 const Eigen::MatrixBase<CovarianceType> &P_init)
  {
    init(x_init, P_init);
  }

  void init(const Eigen::MatrixBase<StateType> &x_init,
            const Eigen::MatrixBase<CovarianceType> &P_init)
  {
    x = x_init;
    P = P_init;
  }

  template <typename ControlSignalType>
  void predict(const Eigen::MatrixBase<ControlSignalType> &u,
               const Eigen::MatrixBase<CovarianceType> &Q,
               const Eigen::MatrixBase<PredictionJacobianType> *F=0)
  {

    if (F != 0)
    {

    }
    else
    {

    }

  }

  template <typename MeasurementFunctor>
  void innovate(const Eigen::MatrixBase<typename MeasurementFunctor::InputType> &measurement,
                const Eigen::MatrixBase<Eigen::Matrix<Scalar, MeasurementFunctor::ValueType::RowsAtCompileTime, MeasurementFunctor::ValueType::RowsAtCompileTime> > &R,
                const Eigen::MatrixBase<Eigen::Matrix<Scalar, MeasurementFunctor::ValueType::RowsAtCompileTime, MeasurementFunctor::InputType::RowsAtCompileTime> > *H=0)
  {

    std::cout << "Prediction!" << std::endl;
    std::cout << "Measurement = " << measurement << std::endl;
    std::cout << "R = " << R << std::endl;

    if (H != 0)
    {
      std::cout << "H = " << H << std::endl;
    }
    else
    {

    }

  }

private:

  StateType x;
  CovarianceType P;

};

}

#endif
