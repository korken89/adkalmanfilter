#pragma once

#include <Eigen/Dense>

namespace ADKalmanFilter
{
/**
 * @brief   Used for defining no outlier rejection.
 */
struct NoOutlierRejection
{
  /**
   * @brief   Required function of the ADKalmanFilter. This version accepts all
   *          measurements.
   *
   * @tparam  Params   List of parameters to ignore.
   */
  template < typename... Params >
  constexpr bool rejectMeasurement(const Params &...) const
  {
    return false;
  }
};

/**
 * @brief   Mahalanobis distance based outlier rejection.
 */
template < int NStdDevs >
struct MahalanobisOutlierRejection
{
  /**
   * @brief   Required function of the ADKalmanFilter. This version accepts a
   *          measurement if the Mahalanobis distance is below a given
   *          threshold.
   *
   * @tparam  SMatrix   Type of the inverted S matrix.
   * @tparam  Residual  Type of the residual.
   */
  template < typename SMatrix, typename Residual >
  constexpr bool rejectMeasurement(const Eigen::MatrixBase< SMatrix > &Sinv,
                                   const Eigen::MatrixBase< Residual > &r) const
  {
    using Scalar = typename Residual::Scalar;

    /* Calculate the Mahalanobis distance: sqrt( r^T * S^-1 * r )
       Simple thresholding for testing measurements. */
    return (Scalar(NStdDevs * NStdDevs) < (r.transpose() * Sinv * r).value());
  }
};
}
