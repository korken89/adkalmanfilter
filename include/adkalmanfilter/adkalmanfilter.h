#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#ifndef _AD_KF
#define _AD_KF

namespace ADKalmanFilter {

/*
 *
 */
template <typename PredictionFunctor>
class ADKalmanFilter
{

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*
   * Static assertions.
   */
  EIGEN_STATIC_ASSERT(PredictionFunctor::InputType::RowsAtCompileTime ==
                      PredictionFunctor::ValueType::RowsAtCompileTime,
            "The PredictionFunctor's input and output must have the same size")

  /*
   * All functors must have the following defined types:
   *
   * Functor::InputType
   * Functor::ValueType
   *
   * We use this to infer state size at compile time.
   */

  typedef typename PredictionFunctor::InputType::Scalar Scalar;
  typedef typename PredictionFunctor::InputType         StateType;
  typedef Eigen::Matrix<Scalar,
                        StateType::RowsAtCompileTime,
                        StateType::RowsAtCompileTime>   StateCovarianceType;
  typedef Eigen::Matrix<Scalar,
                        StateType::RowsAtCompileTime,
                        StateType::RowsAtCompileTime>   PredictionJacobianType;
  typedef Eigen::Matrix<Scalar, 1, 1>                   ArgDefaultType;

  /*
   *
   */
  ADKalmanFilter()
  {
    initialized = false;
  }

  /*
   *
   */
  ADKalmanFilter(const Eigen::MatrixBase<StateType> &x_init,
                 const Eigen::MatrixBase<StateCovarianceType> &P_init)
  {
    init(x_init, P_init);
  }

  /*
   *
   */
  void init(const Eigen::MatrixBase<StateType> &x_init,
            const Eigen::MatrixBase<StateCovarianceType> &P_init)
  {
    x = x_init;
    P = P_init;
    initialized = true;
  }

  /*
   *
   */
  void getState(Eigen::MatrixBase<StateType> &x_out)
  {
    x_out = x;
  }

  /*
   *
   */
  void setState(const Eigen::MatrixBase<StateType> &x_in)
  {
    x = x_in;
  }

  /*
   *
   */
  void getStateCovariance(Eigen::MatrixBase<StateCovarianceType> &P_out)
  {
    P_out = P;
  }

  /*
   *
   */
  void setStateCovariance(const Eigen::MatrixBase<StateCovarianceType> &P_in)
  {
    P = P_in;
  }

  /*
   *
   */
  template <typename ControlSignalType>
  void predict(const Eigen::MatrixBase<StateCovarianceType> &Q,
               const Eigen::MatrixBase<PredictionJacobianType> &F,
               const Eigen::MatrixBase<ControlSignalType> &u)
  {
    StateType f;

    PredictionFunctor()(x, &f, u);
    applyPrediction(f, F, Q);
  }

  /*
   *
   */
  void predict(const Eigen::MatrixBase<StateCovarianceType> &Q,
               const Eigen::MatrixBase<PredictionJacobianType> &F)
  {
    StateType f;

    PredictionFunctor()(x, &f);
    applyPrediction(f, F, Q);
  }

  /*
   *
   */
  template <typename ControlSignalType>
  void predict(const Eigen::MatrixBase<StateCovarianceType> &Q,
               const Eigen::MatrixBase<ControlSignalType> &u)
  {
    StateType f;
    PredictionJacobianType F;
    Eigen::AutoDiffJacobian< PredictionFunctor > adjac;

    adjac(x, &f, &F, u);
    applyPrediction(f, F, Q);
  }

  /*
   *
   */
  void predict(const Eigen::MatrixBase<StateCovarianceType> &Q)
  {
    StateType f;
    PredictionJacobianType F;
    Eigen::AutoDiffJacobian< PredictionFunctor > adjac;

    adjac(x, &f, &F);
    applyPrediction(f, F, Q);
  }

  /*
   *
   */
  void applyPrediction(const Eigen::MatrixBase<StateType> &f,
                       const Eigen::MatrixBase<PredictionJacobianType> &F,
                       const Eigen::MatrixBase<StateCovarianceType> &Q)
  {
    if (!initialized)
      return;

    x += f;
    P = F * P * F.transpose() + Q;
  }

  /*
   *
   */
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType>
  bool update(const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R)
  {
    /*
     * Error checking.
     */
    EIGEN_STATIC_ASSERT(MeasurementFunctor::InputType::RowsAtCompileTime ==
                        PredictionFunctor::InputType::RowsAtCompileTime,
                        "MeasurementFunctor has wrong state size")

    EIGEN_STATIC_ASSERT(MeasurementFunctor::ValueType::RowsAtCompileTime ==
                        MeasurementType::RowsAtCompileTime,
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
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType,
            typename HType>
  bool update(const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R,
              const Eigen::MatrixBase<HType> &H)
  {
    /*
     * Error checking.
     */
    EIGEN_STATIC_ASSERT(MeasurementFunctor::InputType::RowsAtCompileTime ==
                        PredictionFunctor::InputType::RowsAtCompileTime,
                        "MeasurementFunctor has wrong state size");

    EIGEN_STATIC_ASSERT(MeasurementFunctor::ValueType::RowsAtCompileTime ==
                        MeasurementType::RowsAtCompileTime,
                        "MeasurementType has wrong size");

    /*
     * Implementation.
     */
    Eigen::AutoDiffJacobian< MeasurementFunctor > adjac;
    MeasurementType residual, h;

    MeasurementFunctor()(x, &h);
    residual = measurement - h;
    return applyResidual<MeasurementFunctor>(residual, R, H);
  }

  /*
   *
   */
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType,
            typename HType>
  bool applyResidual(const Eigen::MatrixBase<MeasurementType> &residual,
                     const Eigen::MatrixBase<RType> &R,
                     const Eigen::MatrixBase<HType> &H)
  {
    /*
     * Error checking.
     */
    EIGEN_STATIC_ASSERT((std::is_same<Scalar,
                        typename MeasurementFunctor::InputType::Scalar>::value),
            "PredictionFunctor and MeasurementFunctor are of different types")

    EIGEN_STATIC_ASSERT(RType::RowsAtCompileTime ==
                        MeasurementType::RowsAtCompileTime &&
                        RType::ColsAtCompileTime ==
                        MeasurementType::RowsAtCompileTime,
                        "RType has wrong size")

    EIGEN_STATIC_ASSERT(int(HType::RowsAtCompileTime) ==
                        int(MeasurementType::RowsAtCompileTime) &&
                        int(HType::ColsAtCompileTime) ==
                        int(StateType::RowsAtCompileTime),
                        "HType has wrong size")

    if (!initialized)
      return false;

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

private:

  /*
   *
   */
  StateType x;

  /*
   *
   */
  StateCovarianceType P;

  /*
   *
   */
  bool initialized;

};

}

#endif

#include "adkalmanfilter/outlierrejection_impl.h"
