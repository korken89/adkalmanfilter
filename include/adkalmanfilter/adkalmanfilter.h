#include <Eigen/Dense>

#ifndef _AD_KF_H
#define _AD_KF_H

/* Warn users, just in case. */
#ifndef NDEBUG
  #warning "Compiling in Debug mode, all Eigen operations will be very slow"
#endif

namespace ADKalmanFilter {

/**
 * @brief   Base functor for the PredictionFunctor and MeasurementFunctor
 *          to define required types for the AutoDiff module.
 *
 * @tparam    Scalar  Scalar of the code, can be float or double.
 * @tparam    N       Number of rows in the input vector.
 * @tparam    M       Number of rows in the output vector. Defaults to N.
 */
template <typename Scalar, int N, int M=N>
struct BaseFunctor
{
  /*
   * Definitions required for input and output type.
   * Used by the AutoDiff to find Jacobians and ADKalmanFilter to infer sizes.
   */
  typedef Eigen::Matrix<Scalar, N, 1> InputType;
  typedef Eigen::Matrix<Scalar, M, 1> ValueType;
  typedef Eigen::Matrix<Scalar, M, N> JacobianType;
  typedef Eigen::Matrix<Scalar, M, M> CovarianceType;
};


/**
 * @brief   Main class for the AutoDiff Kalman Filter.
 *
 * @details Based on the Eigen template library (http://eigen.tuxfamily.org)
 *          and requires Eigen 3.3. The filter is designed to be very generic
 *          and it utilizes Eigen's AutoDiff module for algorithmic (automatic)
 *          differentiation if no prediction Jacobian (F) or measurement
 *          Jacobian (H) is supplied, while being very fast and efficient.
 *          Generally, using the AutoDiff functionality does not have any real
 *          impact on performance and removes the human factor in manually
 *          generating the Jacobians. The filter supports any number of stand
 *          alone measurement Functors being applied at different rates for
 *          implementing multi sensor fusion. Moreover mismatches in state,
 *          measurement or any other size is detected at compile time.
 *
 * @note    For a tutorial in the usage of the filter, please see the Wiki.
 *
 * @note    This should be compiled with C++11 or later.
 *
 * @tparam  PredictionFunctor   Functor performing the Kalman Filter
 *                              predictions.
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
   * All functors have the following defined types defined in BaseFunctor:
   *
   * Functor::InputType
   * Functor::ValueType
   * Functor::JacobianType
   * Functor::Covariance
   *
   * We use this to infer state size at compile time.
   */

  /**
   * @typedef   Defines the scalar type of the filter from the prediction
   *            Functor's type. The measurement Functor must use this type.
   */
  typedef typename PredictionFunctor::InputType::Scalar Scalar;

  /**
   * @typedef   Defines the state type from the prediction Functor. The
   *            measurement Functor's input type must be the same.
   */
  typedef typename PredictionFunctor::InputType StateType;

  /**
   * @typedef   Uses the state type to infer the size of the state covariance.
   */
  typedef typename PredictionFunctor::CovarianceType StateCovarianceType;

  /**
   * @typedef   Uses the state covariance type to infer the size of the
   *            prediction Jacobian.
   */
  typedef typename PredictionFunctor::JacobianType PredictionJacobianType;

  /**
   * @brief Default constructor.
   */
  ADKalmanFilter();

  /**
   * @brief 	Default constructor which initializes the filter.
   *
   * @tparam xType      State type.
   * @tparam PType      State Covariance type.
   *
   * @param[in] x_init	Starting value for the state.
   * @param[in] P_init  Starting value for the covariance.
   */
  template <typename xType, typename PType>
  ADKalmanFilter(const Eigen::MatrixBase<xType> &x_init,
                 const Eigen::MatrixBase<PType> &P_init);

  /**
   * @brief 	Initializes or reinitializes the filter.
   *
   * @tparam xType      State type.
   * @tparam PType      State Covariance type.
   *
   * @param[in] x_init  Starting value for the state.
   * @param[in] P_init  Starting value for the covariance.
   */
  template <typename xType, typename PType>
  void init(const Eigen::MatrixBase<xType> &x_init,
            const Eigen::MatrixBase<PType> &P_init);

  /**
   * @brief  Getter function for the state.
   *
   * @tparam xType      State type.
   *
   * @param[out] x_out  Output of the state.
   */
  template <typename xType>
  void getState(Eigen::MatrixBase<xType> &x_out);

  /**
   * @brief  Setter function for the state.
   *
   * @tparam xType      State type.
   *
   * @param[in] x_in    Input for the state.
   */
  template <typename xType>
  void setState(const Eigen::MatrixBase<xType> &x_in);

  /**
   * @brief  Getter function for the state covariance.
   *
   * @tparam PType      State Covariance type.
   *
   * @param[out] x_out  Output of the state covariance.
   */
  template <typename PType>
  void getStateCovariance(Eigen::MatrixBase<PType> &P_out);

  /**
   * @brief  Setter function for the state covariance.
   *
   * @tparam PType      State Covariance type.
   *
   * @param[in] x_in    Input for the state covariance.
   */
  template <typename PType>
  void setStateCovariance(const Eigen::MatrixBase<PType> &P_in);

  /**
   * @brief  Returns if the filter has been initialized.
   *
   * @note   The predict and update functions will not perform any action until
   *         after the filter has been initialized.
   *
   * @return True if the filter has been initialized.
   */
  bool isInitialized() const;


  /**
   * @brief  Performs a prediction with a supplied prediction Jacobian.
   *
   * @tparam  FType       Type of the prediction Jacobian.
   * @tparam  QType       Type of the prediction covariance.
   * @tparam  ParamsType  Type list for optional parameters.
   *
   * @param[in] F       Prediction Jacobian input.
   * @param[in] Q       Prediction covariance input.
   * @param[in] params  List of optional parameters and control signals.
   */
  template <typename FType, typename QType, typename... ParamsType>
  void predict(const Eigen::MatrixBase<FType> &F,
               const Eigen::MatrixBase<QType> &Q,
               const ParamsType & ...params);

  /**
   * @brief  Performs a prediction without a supplied prediction Jacobian and
   *         utilizes algorithmic differentiation.
   *
   * @tparam  QType       Type of the prediction covariance.
   * @tparam  ParamsType  Type list for optional parameters.
   *
   * @param[in] Q       Prediction covariance input.
   * @param[in] params  List of optional parameters and control signals.
   */
  template <typename QType, typename... ParamsType>
  void predictAD(const Eigen::MatrixBase<QType> &Q,
                 const ParamsType & ...params);

  /**
   * @brief  Applies a calculated prediction, from one of the predict functions,
   *         to the state and the covariance.
   *
   * @tparam  fType       Type of the prediction state.
   * @tparam  FType       Type of the prediction Jacobian.
   * @tparam  QType       Type of the prediction covariance.
   *
   * @param[in] f   Predicted new state.
   * @param[in] F   Prediction Jacobian.
   * @param[in] Q   Prediction covariance.
   */
  template <typename fType, typename FType, typename QType>
  void applyPrediction(const Eigen::MatrixBase<fType> &f,
                       const Eigen::MatrixBase<FType> &F,
                       const Eigen::MatrixBase<QType> &Q);

  /**
   * @brief  Performs a measurement update of the filter utilizing the supplied
   *         measurement Functor.
   *
   * @note   This utilizes algorithmic differentiation.
   *
   * @tparam  MeasurementFunctor  The supplied measurement Functor.
   * @tparam  MeasurementType     Type of the measurement.
   * @tparam  RType               Type of the measurement covariance.
   * @tparam  ParamsType          Type list for optional parameters.
   *
   * @param[in] measurement   Measurement corresponding to the measurement
   *                          Functor.
   * @param[in] R             Measurement covariance input.
   * @param[in] params        List of optional parameters.
   *
   * @return  Returns true if the measurement was accepted. False indicates
   *          that the outlier rejection rejected the measurement.
   */
  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType,
            typename... ParamsType>
  bool updateAD(const Eigen::MatrixBase<MeasurementType> &measurement,
                const Eigen::MatrixBase<RType> &R,
                const ParamsType & ...params);

  /**
   * @brief  Performs a measurement update of the filter utilizing the supplied
   *         measurement Functor.
   *
   * @tparam  MeasurementFunctor  The supplied measurement Functor.
   * @tparam  HType               Type of the measurement Jacobian.
   * @tparam  MeasurementType     Type of the measurement.
   * @tparam  RType               Type of the measurement covariance.
   * @tparam  ParamsType          Type list for optional parameters.
   *
   * @param[in] H             Measurement Jacobian input.
   * @param[in] measurement   Measurement corresponding to the measurement
   *                          Functor.
   * @param[in] R             Measurement covariance input.
   * @param[in] params        List of optional parameters.
   *
   * @return  Returns true if the measurement was accepted. False indicates
   *          that the outlier rejection rejected the measurement.
   */
  template <typename MeasurementFunctor,
            typename HType,
            typename MeasurementType,
            typename RType,
            typename... ParamsType>
  bool update(const Eigen::MatrixBase<HType> &H,
              const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R,
              const ParamsType & ...params);

  /**
   * @brief  Applies a calculated residual, from one of the update functions,
   *         to the state and the covariance.
   *
   * @tparam  MeasurementFunctor  The supplied measurement Functor.
   * @tparam  HType               Type of the measurement Jacobian.
   * @tparam  MeasurementType     Type of the measurement.
   * @tparam  RType               Type of the measurement covariance.
   *
   * @param[in] H         Measurement Jacobian input.
   * @param[in] residual  Measurement corresponding to the measurement Functor.
   * @param[in] R         Measurement covariance input.
   *
   * @return  Returns true if the measurement was accepted. False indicates
   *          that the outlier rejection rejected the measurement.
   */
  template <typename MeasurementFunctor,
            typename HType,
            typename MeasurementType,
            typename RType>
  bool applyResidual(const Eigen::MatrixBase<HType> &H,
                     const Eigen::MatrixBase<MeasurementType> &residual,
                     const Eigen::MatrixBase<RType> &R);

private:

  /**
   * @var   State variable, holds the current state.
   */
  StateType x;

  /**
   * @var   State covariance variable, holds the current state covariance.
   */
  StateCovarianceType P;

  /**
   * @var   Holder for the initialized flag.
   */
  bool initialized;

};

}

#endif

#include "adkalmanfilter/implementation/outlierrejection_impl.h"
#include "adkalmanfilter/implementation/adkalmanfilter_impl.h"
