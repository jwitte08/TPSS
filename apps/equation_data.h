/*
 * equation_data.h
 *
 *  Created on: Jul 08, 2019
 *      Author: witte
 */

#ifndef EQUATION_DATA_H_
#define EQUATION_DATA_H_

#include <deal.II/base/tensor_function.h>

#include "utilities.h"

using namespace dealii;

namespace Laplace
{
template<int dim>
class SolutionBase
{
protected:
  static const std::size_t n_source_centers = 3;
  static const Point<dim>  source_centers[n_source_centers];
  static const double      width;
};

template<>
const Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] = {Point<1>(0.0),
                                                                                     Point<1>(0.25),
                                                                                     Point<1>(0.6)};

template<>
const Point<2> SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  {Point<2>(0.0, +0.0), Point<2>(0.25, 0.85), Point<2>(+0.6, 0.4)};

template<>
const Point<3> SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] =
  {Point<3>(0.0, 0.0, 0.0), Point<3>(0.25, 0.85, 0.85), Point<3>(0.6, 0.4, 0.4)};

template<int dim>
const double SolutionBase<dim>::width = 1. / 3.;

template<int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
  Solution() : Function<dim>(), SolutionBase<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    double       val   = 0;
    const double pi    = dealii::numbers::PI;
    const double width = SolutionBase<dim>::width;
    for(unsigned int i = 0; i < SolutionBase<dim>::n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - SolutionBase<dim>::source_centers[i];
      val += std::exp(-x_minus_xi.norm_square() / (width * width));
    }
    val /= dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * width);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int = 0) const override final
  {
    dealii::Tensor<1, dim> grad;
    const double           pi    = dealii::numbers::PI;
    const double           width = SolutionBase<dim>::width;
    for(unsigned int i = 0; i < SolutionBase<dim>::n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - SolutionBase<dim>::source_centers[i];
      grad +=
        -2. / (width * width) * std::exp(-x_minus_xi.norm_square() / (width * width)) * x_minus_xi;
    }
    return (grad / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * width));
  }

  virtual double
  laplacian(const dealii::Point<dim> & p, const unsigned int = 0) const override final
  {
    double       lapl  = 0;
    const double pi    = dealii::numbers::PI;
    const double width = SolutionBase<dim>::width;
    for(unsigned int i = 0; i < SolutionBase<dim>::n_source_centers; ++i)
    {
      const auto x_minus_xi = p - SolutionBase<dim>::source_centers[i];
      lapl += (2. / (width * width) * (x_minus_xi * x_minus_xi) - static_cast<double>(dim)) *
              std::exp(-x_minus_xi.norm_square() / (width * width));
    }
    lapl *= 2. / (width * width) / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * width);
    return lapl;
  }
};

constexpr double wave_number = 3.;

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() : Function<dim>()
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    double val = 0;
    val -= solution_function.laplacian(p);
    return val;
  }

private:
  Solution<dim> solution_function;
};
} // end namespace Laplace

namespace LinElasticity
{
struct EquationData
{
  double mu        = 1.;
  double lambda    = 1.;
  double ip_factor = 10.; // required to stabilize discretization !!

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Equation Data:", "");
    oss << Util::parameter_to_fstring("Lame coefficient (mu):", mu);
    oss << Util::parameter_to_fstring("Lame coefficient (lambda):", lambda);
    oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
    return oss.str();
  }
};

template<int dim>
class AnalyticalSolution : public TensorFunction<1, dim>
{
public:
  using value_type          = typename TensorFunction<1, dim>::value_type;
  static constexpr double a = 0.05;

  AnalyticalSolution(const EquationData & equation_data_in)
    : TensorFunction<1, dim>(), equation_data(equation_data_in)
  {
  }

  value_type
  value(const Point<dim> & p) const override final
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    value_type   val;
    const double x  = p[0];
    const double y  = p[1];
    const double pi = numbers::PI;

    val[0] = -a * std::sin(2. * pi * x * x) * std::sin(pi * y);
    val[1] = a * std::sin(pi * x * x) * std::sin(pi * y);

    return val;
  }

private:
  const EquationData equation_data;
};

template<int dim>
class VolumeForce : public TensorFunction<1, dim>
{
public:
  using value_type = typename TensorFunction<1, dim>::value_type;

  VolumeForce(const EquationData & equation_data_in)
    : TensorFunction<1, dim>(), equation_data(equation_data_in)
  {
  }

  value_type
  value(const Point<dim> & p) const override final
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    value_type       val;
    const double     x      = p[0];
    const double     y      = p[1];
    const double     pi     = numbers::PI;
    const double     lambda = equation_data.lambda;
    const double     mu     = equation_data.mu;
    constexpr double a      = AnalyticalSolution<dim>::a;

    // *** -2mu * div.e(u)
    val[0] =
      -1. * a * mu * pi *
      (2. * pi * x * std::cos(pi * x * x) * std::cos(pi * y) +
       (-8. * std::cos(2 * pi * x * x) + pi * (1. + 32. * x * x) * std::sin(2. * pi * x * x)) *
         std::sin(pi * y));
    val[1] =
      2. * a * mu * pi *
      (2. * pi * x * std::cos(2. * pi * x * x) * std::cos(pi * y) +
       (-std::cos(pi * x * x) + pi * (1. + 2. * x * x) * std::sin(pi * x * x)) * std::sin(pi * y));

    // *** -lambda * grad div(u)
    val[0] += (-2.) * a * lambda * pi *
              (pi * x * std::cos(pi * x * x) * std::cos(pi * y) -
               2 * (std::cos(2 * pi * x * x) - 4. * pi * x * x * std::sin(2 * pi * x * x)) *
                 std::sin(pi * y));
    val[1] += a * lambda * pi * pi *
              (4. * x * std::cos(2. * pi * x * x) * std::cos(pi * y) +
               std::sin(pi * x * x) * std::sin(pi * y));

    return val;
  }

private:
  const EquationData equation_data;
};
} // end namespace LinElasticity


#endif /* EQUATION_DATA_H_ */
