#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#ifndef _AD_KF_IMPL_H
#define _AD_KF_IMPL_H

namespace ADKalmanFilter {

/*
 *
 */
template <typename PredictionFunctor>
ADKalmanFilter<PredictionFunctor>::ADKalmanFilter()
{
  initialized = false;
}

/*
 *
 */
template <typename PredictionFunctor>
ADKalmanFilter<PredictionFunctor>::ADKalmanFilter(
              const Eigen::MatrixBase<StateType> &x_init,
              const Eigen::MatrixBase<StateCovarianceType> &P_init)
{
  init(x_init, P_init);
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::init(
              const Eigen::MatrixBase<StateType> &x_init,
              const Eigen::MatrixBase<StateCovarianceType> &P_init)
{
  x = x_init;
  P = P_init.template selfadjointView< Eigen::Upper >();
  initialized = true;
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::getState(
              Eigen::MatrixBase<StateType> &x_out)
{
  x_out = x;
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::setState(
              const Eigen::MatrixBase<StateType> &x_in)
{
  x = x_in;
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::getStateCovariance(
              Eigen::MatrixBase<StateCovarianceType> &P_out)
{
  P_out = P;
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::setStateCovariance(
              const Eigen::MatrixBase<StateCovarianceType> &P_in)
{
  P = P_in.template selfadjointView< Eigen::Upper >();
}

/*
 *
 */
template <typename PredictionFunctor>
bool ADKalmanFilter<PredictionFunctor>::isInitialized() const
{
  return initialized;
}

/*
 *
 */
template <typename PredictionFunctor>
template <typename ControlSignalType>
void ADKalmanFilter<PredictionFunctor>::predict(
              const Eigen::MatrixBase<PredictionJacobianType> &F,
              const Eigen::MatrixBase<ControlSignalType> &u,
              const Eigen::MatrixBase<StateCovarianceType> &Q)
{
  eigen_assert(initialized == true);

  StateType f;

  PredictionFunctor()(x, &f, u);
  applyPrediction(f, F, Q);
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::predict(
              const Eigen::MatrixBase<PredictionJacobianType> &F,
              const Eigen::MatrixBase<StateCovarianceType> &Q)
{
  eigen_assert(initialized == true);

  StateType f;

  PredictionFunctor()(x, &f);
  applyPrediction(f, F, Q);
}

/*
 *
 */
template <typename PredictionFunctor>
template <typename ControlSignalType>
void ADKalmanFilter<PredictionFunctor>::predict(
              const Eigen::MatrixBase<ControlSignalType> &u,
              const Eigen::MatrixBase<StateCovarianceType> &Q)
{
  eigen_assert(initialized == true);

  StateType f;
  PredictionJacobianType F;
  Eigen::AutoDiffJacobian< PredictionFunctor > adjac;

  adjac(x, &f, &F, u);
  applyPrediction(f, F, Q);
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::predict(
              const Eigen::MatrixBase<StateCovarianceType> &Q)
{
  eigen_assert(initialized == true);

  StateType f;
  PredictionJacobianType F;
  Eigen::AutoDiffJacobian< PredictionFunctor > adjac;

  adjac(x, &f, &F);
  applyPrediction(f, F, Q);
}

/*
 *
 */
template <typename PredictionFunctor>
void ADKalmanFilter<PredictionFunctor>::applyPrediction(
              const Eigen::MatrixBase<StateType> &f,
              const Eigen::MatrixBase<PredictionJacobianType> &F,
              const Eigen::MatrixBase<StateCovarianceType> &Q)
{
  x = f;
  P = F * P * F.transpose() + Q;
}

/*
 *
 */
template <typename PredictionFunctor>
template <typename MeasurementFunctor,
          typename MeasurementType,
          typename RType>
bool ADKalmanFilter<PredictionFunctor>::update(
              const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::InputType::RowsAtCompileTime) ==
                      int(PredictionFunctor::InputType::RowsAtCompileTime),
                      "MeasurementFunctor has wrong state size")

  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::ValueType::RowsAtCompileTime) ==
                      int(MeasurementType::RowsAtCompileTime),
                      "MeasurementType has wrong size")

  /*
   * Definitions.
   */
  typedef Eigen::Matrix<Scalar,
                        MeasurementFunctor::ValueType::RowsAtCompileTime,
                        MeasurementFunctor::InputType::RowsAtCompileTime>
                          HType;


  /*
   * Implementation.
   */
  eigen_assert(initialized == true);

  Eigen::AutoDiffJacobian< MeasurementFunctor > adjac;
  HType Had;
  MeasurementType residual, h;

  /* No input Jacobian was given, use Algorithmic Differentiation to
   * calculate the Jacobian. */
  adjac(x, &h, &Had);
  residual = measurement - h;

  return applyResidual<MeasurementFunctor>(residual, R, Had);
}

/*
 *
 */
template <typename PredictionFunctor>
template <typename MeasurementFunctor,
          typename MeasurementType,
          typename RType,
          typename HType>
bool ADKalmanFilter<PredictionFunctor>::update(
              const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R,
              const Eigen::MatrixBase<HType> &H)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::InputType::RowsAtCompileTime) ==
                      int(PredictionFunctor::InputType::RowsAtCompileTime),
                      "MeasurementFunctor has wrong state size");

  EIGEN_STATIC_ASSERT(int(MeasurementFunctor::ValueType::RowsAtCompileTime) ==
                      int(MeasurementType::RowsAtCompileTime),
                      "MeasurementType has wrong size");

  /*
   * Implementation.
   */
  eigen_assert(initialized == true);

  Eigen::AutoDiffJacobian< MeasurementFunctor > adjac;
  MeasurementType residual, h;

  MeasurementFunctor()(x, &h);
  residual = measurement - h;
  return applyResidual<MeasurementFunctor>(residual, R, H);
}

/*
 *
 */
template <typename PredictionFunctor>
template <typename MeasurementFunctor,
          typename MeasurementType,
          typename RType,
          typename HType>
bool ADKalmanFilter<PredictionFunctor>::applyResidual(
              const Eigen::MatrixBase<MeasurementType> &residual,
              const Eigen::MatrixBase<RType> &R,
              const Eigen::MatrixBase<HType> &H)
{
  /*
   * Error checking.
   */
  EIGEN_STATIC_ASSERT((std::is_same<Scalar,
                      typename MeasurementFunctor::InputType::Scalar>::value),
      "PredictionFunctor and MeasurementFunctor are of different Scalar types")

  EIGEN_STATIC_ASSERT(int(RType::RowsAtCompileTime) ==
                      int(MeasurementType::RowsAtCompileTime) &&
                      int(RType::ColsAtCompileTime) ==
                      int(MeasurementType::RowsAtCompileTime),
                      "RType has wrong size")

  EIGEN_STATIC_ASSERT(int(HType::RowsAtCompileTime) ==
                      int(MeasurementType::RowsAtCompileTime) &&
                      int(HType::ColsAtCompileTime) ==
                      int(StateType::RowsAtCompileTime),
                      "HType has wrong size")

  /*
   * Definitions.
   */
  typedef Eigen::Matrix<Scalar,
                        StateType::RowsAtCompileTime,
                        MeasurementType::RowsAtCompileTime>
                        KalmanGainType;

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
    StateCovarianceType KH = StateCovarianceType::Identity() - K*H;

    P = KH * P * KH.transpose() + K * R * K.transpose();
    x += K * residual;

    /* Numerical hack for long term stability. */
    P = Scalar(0.5) * (P + P.transpose());

    return true;
  }
  else
    return false;
}

}

#endif

