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
   * Static assertions.
   */
  EIGEN_STATIC_ASSERT(PredictionFunctor::InputType::RowsAtCompileTime ==
                      PredictionFunctor::ValueType::RowsAtCompileTime,
                      "The PredictionFunctor's input and output must have the \
                      same size")

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
                        StateType::RowsAtCompileTime> StateCovarianceType;
  typedef Eigen::Matrix<Scalar, StateType::RowsAtCompileTime,
                        StateType::RowsAtCompileTime> PredictionJacobianType;

  typedef Eigen::Matrix<Scalar, 1, 1> ArgDefaultType;

  ADKalmanFilter() { }

  ADKalmanFilter(const Eigen::MatrixBase<StateType> &x_init,
                 const Eigen::MatrixBase<StateCovarianceType> &P_init)
  {
    init(x_init, P_init);
  }

  void init(const Eigen::MatrixBase<StateType> &x_init,
            const Eigen::MatrixBase<StateCovarianceType> &P_init)
  {
    x = x_init;
    P = P_init;
  }

  template <typename ControlSignalType = ArgDefaultType>
  void predict(
               const Eigen::MatrixBase<StateCovarianceType> &Q,
               const Eigen::MatrixBase<ControlSignalType> *u=0,
               const Eigen::MatrixBase<PredictionJacobianType> *F=0)
  {

    std::cout << "Prediction!" << std::endl;
    std::cout << "Q = " << std::endl << Q << std::endl << std::endl;

    if (F != 0)
    {

    }
    else
    {

    }

  }

  template <typename MeasurementFunctor,
            typename MeasurementType,
            typename RType,
            typename HType = Eigen::Matrix<Scalar,
                          MeasurementFunctor::ValueType::RowsAtCompileTime,
                          MeasurementFunctor::InputType::RowsAtCompileTime> >
  void update(const Eigen::MatrixBase<MeasurementType> &measurement,
              const Eigen::MatrixBase<RType> &R,
              const Eigen::MatrixBase<HType> *H=0)
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

    EIGEN_STATIC_ASSERT(RType::RowsAtCompileTime ==
                        MeasurementType::RowsAtCompileTime &&
                        RType::ColsAtCompileTime ==
                        MeasurementType::RowsAtCompileTime,
                        "RType has wrong size");

    //EIGEN_STATIC_ASSERT(HType::RowsAtCompileTime ==
    //                    MeasurementFunctor::ValueType::RowsAtCompileTime,
    //                    "HType has wrong size");

    //EIGEN_STATIC_ASSERT(HType::ColsAtCompileTime ==
    //                    MeasurementFunctor::InputType::RowsAtCompileTime,
    //                    "HType has wrong size");

    /*
     * Implementation.
     */

    typename MeasurementFunctor::ValueType h;

    typedef Eigen::Matrix<Scalar,
                          StateType::RowsAtCompileTime,
                          MeasurementFunctor::ValueType::RowsAtCompileTime>
                          KalmanGainType;

    Eigen::AutoDiffJacobian< MeasurementFunctor > adjac;

    /* Temporaries. */
    KalmanGainType K;
    StateCovarianceType KH;
    StateType correction;
    Eigen::LLT<RType> Sllt;

    /* Run measurement prediction and form the Jacobian if it was not
     * provided. */
    if (H != 0)
    {
      /* When calling AutoDiffJacobin without an output jacobian it only
       * executes the MeasurementFunctor. */
      adjac(x, &h);

      Sllt.compute((*H) * P * (*H).transpose() + R);
      K = Sllt.solve((*H) * P);

      /* Update using the positive Joseph form. */
      KH = StateCovarianceType::Identity() - K*(*H);
      P = KH * P * KH.transpose() + K * R * K.transpose();
    }
    else
    {
      HType Had;
      adjac(x, &h, &Had);

      Sllt.compute(Had * P * Had.transpose() + R);
      K = Sllt.solve(Had * P);

      /* Update using the positive Joseph form. */
      KH = StateCovarianceType::Identity() - K*Had;
      P = KH * P * KH.transpose() + K * R * K.transpose();
    }

    correction = K * (measurement - h);
    x += correction;

    /* Numerical hack for long term stability. */
    P = Scalar(0.5) * (P + P.transpose());

  }

private:

  StateType x;
  StateCovarianceType P;

};

}

#endif
