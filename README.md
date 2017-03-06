# Automatic Differentiation Kalman Filter (AD-KF) Library

## List of Contributors

* Emil Fresk (emil.fresk@gmail.com)

Contributions are most welcome!

## Short description

The AD-KF was designed for ease of use while still being very fast and efficient, and to this end it is based on the latest version of the Eigen template library for numerical computations and also utilizing Eigen's AutoDiff module for Automatic/Algorithmic Differentiation to find the prediction and measurement Jacobians if none are supplied. For a primer on AutoDiff the [Wikipedia article](https://en.wikipedia.org/wiki/Automatic_differentiation) covers most of it, but in short it is NOT Numerical Differentiation nor Symbolic Differentiation, rather it applies the chain rule on computations to find derivatives.

### TODO

* Add a way to say if the measurement functor is feasible. Check if the measurement functor should just return a bool.

### General structure

To use the library two `Functors` need to be defined, one `PredictionFunctor` and one (or multiple) `MeasurementFunctor` -- these define the state size, Jacobian sizes etc. for the AD-KF as will be described in the following sections.

### Important function and types

##### BaseFunctor

The `BaseFunctor` is a convenience struct which contains definitions needed by the AutoDiff tools and is defined as:
```C++
/**
 * @brief   Base for the PredictionFunctor and MeasurementFunctor
 *          to define required types for the AutoDiff module.
 *
 * @tparam    Scalar  Scalar of the code, can be float or double.
 * @tparam    N       Number of rows in the input vector (state vector).
 * @tparam    M       Number of rows in the output vector. Defaults to N.
 */
template <typename Scalar, int N, int M=N>
struct BaseFunctor
{ ... }
```
All Functors we implement will inherit from this one.

#### A problem to implement
For an example to implement we will implement the following prediction function:

![alt text](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20p%20%5C%5C%20v%20%5Cend%7Bbmatrix%7D_%7Bk&plus;1%7Ck%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%20%5CDelta%20t%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20p%20%5C%5C%20v%20%5Cend%7Bbmatrix%7D_%7Bk%7Ck%7D)

with the following measurement function:

![alt text](https://latex.codecogs.com/gif.latex?h_%7Bk%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20p%20%5C%5C%20v%20%5Cend%7Bbmatrix%7D_%7Bk&plus;1%7Ck%7D)

It's a simple system that will be no problem to follow. While this is linear, the AD-KF supports arbitrary non-linearities in its measurement and prediction functions.

##### PredictionFunctor

For implementing the prediction outline in the previous section we create a Functor which inherits from BaseFunctor, to define size and types, while the "grunt work" is done by overloading `operator()`:

```C++
template <typename Scalar>
struct predFunctor : public BaseFunctor<Scalar, 2>
{
  template <typename T1, typename T2>
  void operator() (const T1 &input, T2 *output,
                   const Scalar &dt) const
  {
    /* Implementation... */
    T2 &o = *output;

    o(0) = input(0) + dt*input(1);
    o(1) = input(1);
  }
};
```
All Functors follow a very specific form which is:
```C++
void operator() (&input, *output, &parameters)
```
where the parameters can be sample time, control signals etc. This is implemented as a Variadic Template, so an arbitrary amount of parameters can be used.

**Note: The input and output must be templated for the AutoDiff to work! See T1 and T2 in the code example.**

##### MeasurementFunctor

The measurement Functor is implemented in the same way as the prediction Functor, with the difference that the output is of size 1x1, and one more difference: The AD-KF supports arbitrary outlier rejection, but two are implemented 1: `NoOutlierRejection` and 2: `MahalanobisOutlierRejection<NStdDevs>`, in this example we will use the Mahalanobis outlier detector with a threshold of 50 standard deviations.
```C++
template <typename Scalar>
struct measFunctor : public BaseFunctor<Scalar, 2, 1>,
                     public MahalanobisOutlierRejection<50>
{
  template <typename T1, typename T2>
  void operator() (const T1 &input, T2 *output) const
  {
    /* Implementation... */
    T2 &o = *output;
    o(0) = input(0);
  }
};
```

##### Using these Functors with the AD-KF
Now to get the complete picture, lets implement the full filter (for details see *adkf_test.cpp*):
```C++
/* Create the AD-KF object. */
ADKalmanFilter< predFunctor<double> > kf;

/* Initialize */
kf.init(starting_state, starting_covariance);

/* Predict */
kf.predictAD(process_covariance, dt); // Prediction using the AutoDiff
kf.predict(prediction_jacobian, process_covariance, dt); // Prediction NOT using the AutoDiff

/* Update */
bool accepted = kf.updateAD< measFunctor<double> >(measurement,
                                                   measurement_covariance); // Update using the AutoDiff
bool accepted = kf.update< measFunctor<double> >(measurement_jacobian,
                                                 measurement,
                                                 measurement_covariance); // Update NOT using the AutoDiff

/* accepted indicates if the measurement was accepted by the outlier rejector. */
```

For full details on how to use the filter, have a look at *adkf_test.cpp*.

## Running unit tests

The following will run a simple test on functionality and run over a small dataset.

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `make test`
