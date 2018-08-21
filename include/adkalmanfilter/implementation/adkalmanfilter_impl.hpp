#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#pragma once

namespace ADKalmanFilter
{
/*
 *
 */
template < typename Functor, typename InputType, typename JType,
           typename... ParamsType >
void getJacobianAD(const Eigen::MatrixBase< InputType > &input,
                   Eigen::MatrixBase< JType > &J, const ParamsType &... params)
{
  typename Functor::ValueType f;
  typename Functor::JacobianType JJ;
  Eigen::AutoDiffJacobian< Functor > adjac;

  /* Calculate and return the Jacobian at the value of the input vector. */
  adjac(input, &f, &JJ, params...);
  J = JJ;
}

/*
 *
 */
template < typename PredictionFunctor >
ADKalmanFilter< PredictionFunctor >::ADKalmanFilter()
{
  initialized = false;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename xType, typename PType >
ADKalmanFilter< PredictionFunctor >::ADKalmanFilter(
    const Eigen::MatrixBase< xType > &x_init,
    const Eigen::MatrixBase< PType > &P_init)
{
  init(x_init, P_init);
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename xType, typename PType >
void ADKalmanFilter< PredictionFunctor >::init(
    const Eigen::MatrixBase< xType > &x_init,
    const Eigen::MatrixBase< PType > &P_init)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(xType, StateType)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(PType, StateCovarianceType)

  x = x_init;
  P = P_init;
  initialized = true;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename xType >
void ADKalmanFilter< PredictionFunctor >::getState(
    Eigen::MatrixBase< xType > &x_out)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(xType, StateType)

  x_out = x;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename xType >
void ADKalmanFilter< PredictionFunctor >::setState(
    const Eigen::MatrixBase< xType > &x_in)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(xType, StateType)

  x = x_in;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename PType >
void ADKalmanFilter< PredictionFunctor >::getStateCovariance(
    Eigen::MatrixBase< PType > &P_out)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(PType, StateCovarianceType)

  P_out = P;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename PType >
void ADKalmanFilter< PredictionFunctor >::setStateCovariance(
    const Eigen::MatrixBase< PType > &P_in)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(PType, StateCovarianceType)

  P = P_in;
}

/*
 *
 */
template < typename PredictionFunctor >
bool ADKalmanFilter< PredictionFunctor >::isInitialized() const
{
  return initialized;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename FType, typename QType, typename... ParamsType >
void ADKalmanFilter< PredictionFunctor >::predict(
    const Eigen::MatrixBase< FType > &F, const Eigen::MatrixBase< QType > &Q,
    const ParamsType &... params)
{
  /*
   * Error checking.
   */
  eigen_assert(initialized == true);

  /*
   * Implementation.
   */
  StateType f;

  PredictionFunctor()(x, &f, params...);
  applyPrediction(f, F, Q);
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename QType, typename... ParamsType >
void ADKalmanFilter< PredictionFunctor >::predictAD(
    const Eigen::MatrixBase< QType > &Q, const ParamsType &... params)
{
  /*
   * Error checking.
   */
  eigen_assert(initialized == true);

  /*
   * Implementation.
   */
  StateType f;
  PredictionJacobianType F;
  Eigen::AutoDiffJacobian< PredictionFunctor > adjac;

  adjac(x, &f, &F, params...);
  applyPrediction(f, F, Q);
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename fType, typename FType, typename QType >
void ADKalmanFilter< PredictionFunctor >::applyPrediction(
    const Eigen::MatrixBase< fType > &f, const Eigen::MatrixBase< FType > &F,
    const Eigen::MatrixBase< QType > &Q)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT(
      int(fType::RowsAtCompileTime) == int(StateType::RowsAtCompileTime) &&
          int(fType::ColsAtCompileTime) == 1,
      "fType has wrong size")
  EIGEN_STATIC_ASSERT(int(FType::RowsAtCompileTime) ==
                              int(StateCovarianceType::RowsAtCompileTime) &&
                          int(FType::ColsAtCompileTime) ==
                              int(StateCovarianceType::RowsAtCompileTime),
                      "FType has wrong size")
  EIGEN_STATIC_ASSERT(int(QType::RowsAtCompileTime) ==
                              int(StateCovarianceType::RowsAtCompileTime) &&
                          int(QType::ColsAtCompileTime) ==
                              int(StateCovarianceType::RowsAtCompileTime),
                      "QType has wrong size")

  /*
   * Implementation.
   */
  x = f;
  P = F * P * F.transpose() + Q;
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename MeasurementFunctor, typename MeasurementType,
           typename RType, typename... ParamsType >
bool ADKalmanFilter< PredictionFunctor >::updateAD(
    const Eigen::MatrixBase< MeasurementType > &measurement,
    const Eigen::MatrixBase< RType > &R, const ParamsType &... params)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::InputType::RowsAtCompileTime) ==
                          int(PredictionFunctor::InputType::RowsAtCompileTime),
                      "MeasurementFunctor has wrong state size")

  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::ValueType::RowsAtCompileTime) ==
                              int(MeasurementType::RowsAtCompileTime) &&
                          int(MeasurementType::ColsAtCompileTime) == 1,
                      "MeasurementType has wrong size");

  /*
   * Definitions.
   */
  using HType =
      Eigen::Matrix< Scalar, MeasurementFunctor::ValueType::RowsAtCompileTime,
                     MeasurementFunctor::InputType::RowsAtCompileTime >;

  /*
   * Implementation.
   */
  eigen_assert(initialized == true);

  Eigen::AutoDiffJacobian< MeasurementFunctor > adjac;
  HType Had;
  MeasurementType residual, h;

  /* No input Jacobian was given, use Algorithmic Differentiation to
   * calculate the Jacobian. */
  adjac(x, &h, &Had, params...);
  residual = measurement - h;

  return applyResidual< MeasurementFunctor >(Had, residual, R);
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename MeasurementFunctor, typename HType,
           typename MeasurementType, typename RType, typename... ParamsType >
bool ADKalmanFilter< PredictionFunctor >::update(
    const Eigen::MatrixBase< HType > &H,
    const Eigen::MatrixBase< MeasurementType > &measurement,
    const Eigen::MatrixBase< RType > &R, const ParamsType &... params)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::InputType::RowsAtCompileTime) ==
                          int(PredictionFunctor::InputType::RowsAtCompileTime),
                      "MeasurementFunctor has wrong state size");

  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::ValueType::RowsAtCompileTime) ==
                              int(MeasurementType::RowsAtCompileTime) &&
                          int(MeasurementType::ColsAtCompileTime) == 1,
                      "MeasurementType has wrong size");

  /*
   * Implementation.
   */
  eigen_assert(initialized == true);

  Eigen::AutoDiffJacobian< MeasurementFunctor > adjac;
  MeasurementType residual, h;

  MeasurementFunctor()(x, &h, params...);
  residual = measurement - h;
  return applyResidual< MeasurementFunctor >(H, residual, R);
}

/*
 *
 */
template < typename PredictionFunctor >
template < typename MeasurementFunctor, typename HType,
           typename MeasurementType, typename RType >
bool ADKalmanFilter< PredictionFunctor >::applyResidual(
    const Eigen::MatrixBase< HType > &H,
    const Eigen::MatrixBase< MeasurementType > &residual,
    const Eigen::MatrixBase< RType > &R)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT(
      (std::is_same< Scalar,
                     typename MeasurementFunctor::InputType::Scalar >::value),
      "PredictionFunctor and MeasurementFunctor are of different Scalar types")

  EIGEN_STATIC_ASSERT(
      int(HType::RowsAtCompileTime) ==
              int(MeasurementType::RowsAtCompileTime) &&
          int(HType::ColsAtCompileTime) == int(StateType::RowsAtCompileTime),
      "HType has wrong size")

  EIGEN_STATIC_ASSERT(
      int(RType::RowsAtCompileTime) ==
              int(MeasurementFunctor::ValueType::RowsAtCompileTime) &&
          int(RType::ColsAtCompileTime) ==
              int(MeasurementFunctor::ValueType::RowsAtCompileTime),
      "RType has wrong size")

  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::ValueType::RowsAtCompileTime) ==
                              int(MeasurementType::RowsAtCompileTime) &&
                          int(MeasurementType::ColsAtCompileTime) == 1,
                      "residual has wrong size");

  /*
   * Definitions.
   */
  using KalmanGainType = Eigen::Matrix< Scalar, StateType::RowsAtCompileTime,
                                        MeasurementType::RowsAtCompileTime >;

  /*
   * Implementation.
   */

  /* Form inverse of S for future use in the outlier rejection and
   * in the Kalman Filter equations. */
  RType Sinv = (H * P * H.transpose() + R).inverse();
  /* Testing showed this to be faster than using Eigen::LLT
   * TODO: Figure out why... */

  /* Apply outlier rejection before we do all the heavy calculations. */
  if (MeasurementFunctor().rejectMeasurement(Sinv, residual) == false)
  {
    /* Update using the positive Joseph form. */
    KalmanGainType K = P * H.transpose() * Sinv;
    StateCovarianceType KH = StateCovarianceType::Identity() - K * H;

    P = KH * P * KH.transpose() + K * R * K.transpose();
    x += K * residual;

    /* Numerical hack for long term stability. */
    P = Scalar(0.5) * (P + P.transpose());

    return true;
  }
  else
  {
    return false;
  }
}
}
