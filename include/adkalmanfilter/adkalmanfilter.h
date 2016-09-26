#include <Eigen/Dense>

#ifndef _AD_KF_H
#define _AD_KF_H

/* Warn users, just in case. */
#ifndef NDEBUG
  #warning "Compiling in Debug mode, all Eigen operations will be very slow"
#endif

namespace ADKalmanFilter {

/*
 *
 */
template <typename Scalar, int N, int M=N>
struct BaseFunctor
{
  /*
   * Definitions required for input and output type.
   * Used by the AutoDiff to find Jacobians.
   */
  typedef Eigen::Matrix<Scalar, N, 1> InputType;
  typedef Eigen::Matrix<Scalar, M, 1> ValueType;
};


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
   * All functors have the following defined types:
   *
   * Functor::InputType
   * Functor::ValueType
   *
   * We use this to infer state size at compile time.
   */

  typedef typename PredictionFunctor::InputType::Scalar  Scalar;
  typedef typename PredictionFunctor::InputType          StateType;
  typedef Eigen::Matrix<Scalar,
                        StateType::RowsAtCompileTime,
                        StateType::RowsAtCompileTime>    StateCovarianceType;
  typedef StateCovarianceType                            PredictionJacobianType;

  /*
   *
   */
  ADKalmanFilter();

  /*
   *
   */
  ADKalmanFilter(const Eigen::MatrixBase<StateType> &x_init,
                 const Eigen::MatrixBase<StateCovarianceType> &P_init);

  /*
   *
   */
  void init(const Eigen::MatrixBase<StateType> &x_init,
            const Eigen::MatrixBase<StateCovarianceType> &P_init);

  /*
   *
   */
  void getState(Eigen::MatrixBase<StateType> &x_out);

  /*
   *
   */
  void setState(const Eigen::MatrixBase<StateType> &x_in);

  /*
   *
   */
  void getStateCovariance(Eigen::MatrixBase<StateCovarianceType> &P_out);

  /*
   *
   */
  void setStateCovariance(const Eigen::MatrixBase<StateCovarianceType> &P_in);

  /*
   *
   */
  bool isInitialized() const;

  /*
   *
   */
  template <typename ControlSignalType>

  void predict(const Eigen::MatrixBase<PredictionJacobianType> &F,
               const Eigen::MatrixBase<ControlSignalType> &u,
               const Eigen::MatrixBase<StateCovarianceType> &Q);

  /*
   *
   */
  void predict(const Eigen::MatrixBase<PredictionJacobianType> &F,
               const Eigen::MatrixBase<StateCovarianceType> &Q);


  /*
   *
   */
  template <typename ControlSignalType>

  void predict(const Eigen::MatrixBase<ControlSignalType> &u,
               const Eigen::MatrixBase<StateCovarianceType> &Q);


  /*
   *
   */
  void predict(const Eigen::MatrixBase<StateCovarianceType> &Q);

  /*
   *
   */
  void applyPrediction(const Eigen::MatrixBase<StateType> &f,
                       const Eigen::MatrixBase<PredictionJacobianType> &F,
                       const Eigen::MatrixBase<StateCovarianceType> &Q);

  /*
   *
   */
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType>

  bool update(const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R);

  /*
   *
   */
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType,
            typename HType>

  bool update(const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R,
              const Eigen::MatrixBase<HType> &H);

  /*
   *
   */
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType,
            typename HType>

  bool applyResidual(const Eigen::MatrixBase<MeasurementType> &residual,
                     const Eigen::MatrixBase<RType> &R,
                     const Eigen::MatrixBase<HType> &H);

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

#include "adkalmanfilter/implementation/outlierrejection_impl.h"
#include "adkalmanfilter/implementation/adkalmanfilter_impl.h"
