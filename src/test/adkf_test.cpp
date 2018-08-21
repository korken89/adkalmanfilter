#include <Eigen/Dense>
#include <fstream>
#include "adkalmanfilter/adkalmanfilter.hpp"
#include "gtest/gtest.h"

/*
 * Settings for the test.
 * If changed to float, EPS needs to be increased to about 1e-4.
 */
#define EPS 1e-10
typedef double TestScalar;

/*
 * Functors for the KF tests.
 */
template <typename Scalar>
struct predFunctor : public ADKalmanFilter::BaseFunctor<Scalar, 2> {
  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2>
  void operator()(const T1 &input, T2 *output, const Scalar &dt) const
  {
    /* Implementation... */
    T2 &o = *output;

    auto i0 = input(0);
    auto i1 = input(1);

    o(0) = i0 + dt * i1;
    o(1) = i1;
  }
};

template <typename Scalar>
struct measFunctor : public ADKalmanFilter::BaseFunctor<Scalar, 2, 1>,
                     public ADKalmanFilter::MahalanobisOutlierRejection<50> {
  /*
   * Implementation starts here.
   */
  template <typename T1, typename T2>
  void operator()(const T1 &input, T2 *output) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0);
  }
};

/*
 * Typedefs for ease of use.
 */
typedef Eigen::Matrix<TestScalar, 2, 1> StateType;
typedef Eigen::Matrix<TestScalar, 1, 1> MeasType;
typedef Eigen::Matrix<TestScalar, 1, 1> RType;
typedef Eigen::Matrix<TestScalar, 2, 2> QType;
typedef Eigen::Matrix<TestScalar, 1, 2> HType;
typedef Eigen::Matrix<TestScalar, 2, 2> FType;

/* Sampling time. */
const TestScalar dt = 1e-2;

/* Kalman Filter instantiation. */
ADKalmanFilter::ADKalmanFilter<predFunctor<TestScalar>> kf;
StateType out;
QType P;

TEST(ADKFTest, TestAutoDiff)
{
  /* Test the AutoDiff. */
  StateType x0 = StateType::Zero();
  HType H_expect = HType::Identity();
  HType H_out;
  FType F_expect = (FType() << 1, dt, 0, 1).finished();
  FType F_out;

  /* Calculate and return the Jacobian at the value of the input vector. */
  ADKalmanFilter::getJacobianAD<predFunctor<TestScalar>>(x0, F_out, dt);
  ADKalmanFilter::getJacobianAD<measFunctor<TestScalar>>(x0, H_out);

  ASSERT_NEAR(F_expect(0, 0), F_out(0, 0), EPS);
  ASSERT_NEAR(F_expect(0, 1), F_out(0, 1), EPS);
  ASSERT_NEAR(F_expect(1, 0), F_out(1, 0), EPS);
  ASSERT_NEAR(F_expect(1, 1), F_out(1, 1), EPS);

  ASSERT_NEAR(H_expect(0, 0), H_out(0, 0), EPS);
  ASSERT_NEAR(H_expect(0, 1), H_out(0, 1), EPS);
}

TEST(ADKFTest, TestInitialization)
{
  StateType out;
  QType P;

  /* Initialize the filter. */
  StateType x0 = StateType::Zero();
  QType P0 = (QType() << 10, 0, 0, 0).finished();

  ASSERT_EQ(false, kf.isInitialized());
  kf.init(x0, P0);
  ASSERT_EQ(true, kf.isInitialized());

  kf.getState(out);
  kf.getStateCovariance(P);

  ASSERT_NEAR(P0(0, 0), P(0, 0), EPS);
  ASSERT_NEAR(P0(0, 1), P(0, 1), EPS);
  ASSERT_NEAR(P0(1, 0), P(1, 0), EPS);
  ASSERT_NEAR(P0(1, 1), P(1, 1), EPS);

  ASSERT_NEAR(x0(0), out(0), EPS);
  ASSERT_NEAR(x0(1), out(1), EPS);
}

TEST(ADKFTest, TestRangeDatasetWithOutlierRejection)
{
  std::ifstream rangefile("../test_data/distance_with_outliers.csv");

  TestScalar range, p_out, v_out;
  bool is_outlier, accepted;
  StateType x;
  QType P;

  MeasType meas;
  RType R = (RType() << 0.0016).finished();
  QType Q = (QType() << 0, 0, 0, 1e-2).finished();
  QType Pend = (QType() << 0.000320889946971756, 0.00357646478842555,
                0.00357646478842555, 0.0897226644867756)
                   .finished();

  while (rangefile >> range >> is_outlier >> p_out >> v_out) {
    /* Run the filter over the test data. */
    meas << range;
    kf.predictAD(Q, dt);
    accepted = kf.updateAD<measFunctor<TestScalar>>(meas, R);

    kf.getState(x);
    ASSERT_NEAR(p_out, x(0), EPS);
    ASSERT_NEAR(v_out, x(1), EPS);
    ASSERT_EQ(is_outlier, !accepted);
  }

  kf.getStateCovariance(P);

  ASSERT_NEAR(Pend(0, 0), P(0, 0), EPS);
  ASSERT_NEAR(Pend(0, 1), P(0, 1), EPS);
  ASSERT_NEAR(Pend(1, 0), P(1, 0), EPS);
  ASSERT_NEAR(Pend(1, 1), P(1, 1), EPS);
}

int main(int argc, char *argv[])
{
  std::srand((unsigned int)time(0));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
