/*
 * equation_data.h
 *
 *  Created on: Jul 08, 2019
 *      Author: witte
 */

#ifndef EQUATION_DATA_H_
#define EQUATION_DATA_H_

#include <deal.II/base/tensor_function.h>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

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

struct ZeroDirichletUnitCubeData
{
  static constexpr double a = 1.;
};

template<int dim>
class ZeroDirichletUnitCube : public Function<dim>, private ZeroDirichletUnitCubeData
{
};

template<>
class ZeroDirichletUnitCube<2> : public Function<2>, private ZeroDirichletUnitCubeData
{
private:
  static constexpr int dim = 2;
  using value_type         = Point<dim>::value_type;

public:
  virtual value_type
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    const value_type pi    = dealii::numbers::PI;
    const auto &     x     = p[0];
    const auto &     y     = p[1];
    const auto &     exp_y = std::exp(y * (y - 1.) * (y - 1.));
    value_type       val   = a * std::sin(pi * x * x) * (exp_y - 1.);
    return val;
  }

  virtual value_type
  laplacian(const dealii::Point<dim> & p, const unsigned int = 0) const override final
  {
    const value_type pi    = dealii::numbers::PI;
    const auto &     x     = p[0];
    const auto &     y     = p[1];
    const auto &     exp_y = std::exp(y * (y - 1.) * (y - 1.));

    value_type lapl = 4. * pi * pi * x * x;
    lapl +=
      (-3. - 4. * pi * pi * x * x - 2. * y + 22. * y * y - 24. * y * y * y + 9 * y * y * y * y) *
      exp_y;
    lapl *= std::sin(pi * x * x);
    lapl += 2. * pi * (exp_y - 1.) * std::cos(pi * x * x);
    lapl *= a;
    return lapl;
  }
};

template<>
class ZeroDirichletUnitCube<3> : public Function<3>, private ZeroDirichletUnitCubeData
{
private:
  static constexpr int dim = 3;
  using value_type         = Point<dim>::value_type;
  const ZeroDirichletUnitCube<2> func_2d;

public:
  virtual value_type
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    const value_type pi = dealii::numbers::PI;
    const auto &     x  = p[0];
    const auto &     y  = p[1];
    const auto &     z  = p[2];
    const Point<2>   p_xy(x, y);
    const auto &     val_xy = func_2d.value(p_xy) / a;
    value_type       val    = a * val_xy * std::sin(pi * z) * std::sin(pi * z);
    return val;
  }

  virtual value_type
  laplacian(const dealii::Point<dim> & p, const unsigned int = 0) const override final
  {
    const value_type pi = dealii::numbers::PI;
    const auto &     x  = p[0];
    const auto &     y  = p[1];
    const auto &     z  = p[2];
    const Point<2>   p_xy(x, y);
    const auto &     lapl_xy = func_2d.laplacian(p_xy) / a;
    const auto &     exp_y   = std::exp(y * (y - 1.) * (y - 1.));

    value_type lapl = -lapl_xy + (-1 + exp_y) * 2. * pi * pi * std::sin(pi * x * x);
    lapl *= -std::sin(pi * z) * std::sin(pi * z);
    lapl +=
      2. * (-1 + exp_y) * pi * pi * std::cos(pi * z) * std::cos(pi * z) * std::sin(pi * x * x);
    lapl *= a;
    return lapl;
  }
};

template<int dim>
class ManufacturedLoad : public Function<dim>
{
public:
  ManufacturedLoad(const std::shared_ptr<const Function<dim>> solution_function_in)
    : Function<dim>(), solution_function(solution_function_in)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    return -solution_function->laplacian(p);
  }

private:
  std::shared_ptr<const Function<dim>> solution_function;
};

template<int dim>
class RandomLoad : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &, const unsigned int) const override final
  {
    return make_random_value<double>();
  }
};

} // end namespace Laplace

namespace LinElasticity
{
struct EquationData
{
  enum class PenaltyVariant
  {
    basic,
    tensor
  };

  double         mu             = 1.;
  double         lambda         = 1.;
  PenaltyVariant ip_variant     = PenaltyVariant::basic;
  double         ip_factor      = 10.; // required to stabilize discretization !!
  int            lambda_rank    = -1.;
  int            kronecker_rank = 2.;
  double         factor         = 1.;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Equation Data:", "");
    oss << Util::parameter_to_fstring("Lame coefficient (mu):", mu);
    oss << Util::parameter_to_fstring("Lame coefficient (lambda):", lambda);
    oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
    oss << Util::parameter_to_fstring("Lambda Kronecker rank:", lambda_rank);
    oss << Util::parameter_to_fstring("Schur Kronecker rank:", kronecker_rank);
    oss << Util::parameter_to_fstring("Schur factor ...:", factor);
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

template<int dim>
class VolumeForceRandom : public TensorFunction<1, dim>
{
public:
  Tensor<1, dim>
  value(const Point<dim> &) const override final
  {
    Tensor<1, dim> val;
    for(auto d = 0U; d < dim; ++d)
      val[d] = abs(make_random_value<double>());
    return val;
  }
};

} // end namespace LinElasticity


#endif /* EQUATION_DATA_H_ */
