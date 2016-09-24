#ifndef _AD_KF_OUTLIER_REJECTION_IMPL
#define _AD_KF_OUTLIER_REJECTION_IMPL

namespace ADKalmanFilter {

/*
 * Used for defining no outlier rejection.
 */
struct NoOutlierRejection
{
  template <typename... Params>
  constexpr bool rejectMeasurement(const Params &...) const
  {
    return false;
  }
};


/*
 * Mahalanobis distance based outlier rejection.
 */
template <int NStdDevs>
struct MahalanobisOutlierRejection
{
  template <typename SMatrix, typename Residual>
  constexpr bool rejectMeasurement(const Eigen::MatrixBase<SMatrix> &Sinv,
                                   const Eigen::MatrixBase<Residual> &r) const
  {
    typedef typename Residual::Scalar Scalar;

    /* Calculate the Mahalanobis distance: sqrt( r^T * S^-1 * r )
       Simple thresholding for testing measurements. */
    return (Scalar(NStdDevs*NStdDevs) < (r.transpose() * Sinv * r).value());
  }
};

}

#endif
