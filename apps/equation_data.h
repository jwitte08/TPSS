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



/**
 *
 * PDE data and reference solutions:
 *
 * 1) EquationData
 * 2) Solution & RightHandSide (multivariate Gaussian bell curves)
 * 3) ZeroDirichletUnitCube & ManufacturedLoad (trigonometric)
 * 4) RandomLoad
 *
 */
namespace Laplace
{
struct EquationData
{
  std::set<types::boundary_id> dirichlet_boundary_ids = {0};
  double                       ip_factor              = 1.;
};


template<int dim>
class SolutionBase
{
protected:
  static constexpr std::size_t n_source_centers = 3;
  static const Point<dim>      source_centers[n_source_centers];
  static const double          width;
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


using dealii::numbers::PI;

template<int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
  using SolutionBase<dim>::width;
  using SolutionBase<dim>::n_source_centers;

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    double val = 0;
    for(unsigned int i = 0; i < n_source_centers; ++i)
      val += value_impl(p, i);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int = 0) const override final
  {
    dealii::Tensor<1, dim> grad;
    for(unsigned int i = 0; i < n_source_centers; ++i)
      grad += gradient_impl(p, i);
    return grad;
  }

  virtual double
  laplacian(const dealii::Point<dim> & p, const unsigned int = 0) const override final
  {
    double lapl = 0;
    for(unsigned int i = 0; i < n_source_centers; ++i)
      lapl += laplacian_impl(p, i);
    return lapl;
  }

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int /*component*/ = 0) const override final
  {
    SymmetricTensor<2, dim> hess;
    for(unsigned int i = 0; i < n_source_centers; ++i)
      hess += hessian_impl(p, i);
    return hess;
  }

  double
  bilaplacian(const dealii::Point<dim> & p, const unsigned int = 0) const
  {
    double bilapl = 0;
    for(unsigned int i = 0; i < n_source_centers; ++i)
      bilapl += bilaplacian_impl(p, i);
    return bilapl;
  }

private:
  constexpr double
  u0() const
  {
    return 1. / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * PI) * width);
  }

  constexpr double
  v0() const
  {
    return -1. / (width * width);
  }

  double
  bell(const Point<dim> & p, const unsigned int i) const
  {
    AssertIndexRange(i, n_source_centers);
    const dealii::Tensor<1, dim> x_minus_xi = p - SolutionBase<dim>::source_centers[i];
    return std::exp(v0() * x_minus_xi.norm_square());
  }

  double
  value_impl(const Point<dim> & p, const unsigned int i) const
  {
    return u0() * bell(p, i);
  }

  Tensor<1, dim>
  gradient_impl(const Point<dim> & p, const unsigned int i) const
  {
    const dealii::Tensor<1, dim> x_minus_xi = p - SolutionBase<dim>::source_centers[i];
    return u0() * 2. * v0() * bell(p, i) * x_minus_xi;
  }

  double
  laplacian_impl(const dealii::Point<dim> & p, const unsigned int i) const
  {
    double     lapl       = 0;
    const auto x_minus_xi = p - SolutionBase<dim>::source_centers[i];
    lapl += bell(p, i) * (static_cast<double>(dim) + 2. * v0() * x_minus_xi.norm_square());
    lapl *= u0() * 2. * v0();
    return lapl;
  }

  SymmetricTensor<2, dim>
  hessian_impl(const Point<dim> & p, const unsigned int i) const
  {
    const auto              x_minus_xi = p - SolutionBase<dim>::source_centers[i];
    SymmetricTensor<2, dim> hess;
    for(auto d = 0U; d < dim; ++d)
      hess[d][d] = 2. * u0() * v0() * bell(p, i) * (1. + 2. * v0() * x_minus_xi[d] * x_minus_xi[d]);
    for(auto d1 = 0U; d1 < dim; ++d1)
      for(auto d2 = d1 + 1; d2 < dim; ++d2)
        hess[d1][d2] = 2. * 2. * u0() * v0() * v0() * bell(p, i) * x_minus_xi[d1] * x_minus_xi[d2];
    return hess;
  }

  double
  bilaplacian_impl(const dealii::Point<dim> & p, const unsigned int i) const
  {
    double           bilapl     = 0;
    constexpr double d          = dim;
    const auto       x_minus_xi = p - SolutionBase<dim>::source_centers[i];
    bilapl += d + 2. + v0() * x_minus_xi.norm_square();
    bilapl *= 2. * 2. * v0() * x_minus_xi.norm_square();
    bilapl += d * d + 2. * d;
    bilapl *= 2. * 2. * v0() * v0() * u0() * bell(p, i);
    return bilapl;
  }
};



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



/**
 *
 * PDE data and reference solutions:
 *
 * 1) EquationData
 * 2) AnalyticalSolution & VolumeForce (manufactured)
 * 3) VolumeForceRandom
 *
 */
namespace LinElasticity
{
struct EquationData
{
  enum class PenaltyVariant
  {
    basic,
    tensor
  };
  static std::string
  str_penalty_variant(const PenaltyVariant variant)
  {
    std::string str[] = {"basic", "tensor-driven + lambda-scaled-strain"};
    return str[static_cast<int>(variant)];
  }

  enum class IntegratorVariant
  {
    both,
    strain,
    graddiv
  };
  static std::string
  str_integrator_variant(const IntegratorVariant variant)
  {
    std::string str[] = {"strain + grad-div", "strain", "grad-div"};
    return str[static_cast<int>(variant)];
  }

  double            mu                 = 1.;
  double            lambda             = 1.;
  PenaltyVariant    ip_variant         = PenaltyVariant::basic;
  double            ip_factor          = 10.; // required to stabilize discretization !!
  int               lambda_rank        = -1.;
  int               kronecker_rank     = 2.;
  double            factor             = 1.;
  IntegratorVariant integrator_variant = IntegratorVariant::both;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Equation Data:", "");
    oss << Util::parameter_to_fstring("Choice of integrators (tests only):",
                                      str_integrator_variant(integrator_variant));
    oss << Util::parameter_to_fstring("Lame coefficient (mu):", mu);
    oss << Util::parameter_to_fstring("Lame coefficient (lambda):", lambda);
    oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
    oss << Util::parameter_to_fstring("IP variant:", str_penalty_variant(ip_variant));
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



/**
 *
 * PDE Data and manufactured solutions
 *
 * 1) EquationData
 * 2) ZeroBoundary & Load (step-47)
 * 3) GaussianBells & Load (namespace Laplace)
 *
 */
namespace Biharmonic
{
struct EquationData
{
  enum LocalSolverVariant
  {
    Exact,
    Bilaplacian
  };

  static std::string
  str_local_solver(const LocalSolverVariant variant);

  std::string
  str_local_solver() const
  {
    return str_local_solver(local_solver_variant);
  }

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Equation Data:", "");
    oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
    oss << Util::parameter_to_fstring("Local solver:", str_local_solver(local_solver_variant));
    return oss.str();
  }

  std::set<types::boundary_id> dirichlet_boundary_ids = {0};
  double                       ip_factor              = 1.;
  LocalSolverVariant           local_solver_variant   = LocalSolverVariant::Exact;
};


std::string
EquationData::str_local_solver(const LocalSolverVariant variant)
{
  const std::string str_variant[] = {"Exact", "Bilaplacian (no mixed derivatives)"};
  return str_variant[(int)variant];
}



template<int dim, typename FunctionType>
class ManufacturedLoad : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const override final
  {
    return solution.bilaplacian(p);
  }

private:
  const FunctionType solution;
};



namespace ZeroBoundary
{
using numbers::PI;

template<int dim>
class Solution : public Function<dim>
{
public:
  static_assert(dim == 2, "Only dim==2 is implemented.");

  virtual double
  value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    return std::sin(PI * p[0]) * std::sin(PI * p[1]);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> r;
    r[0] = PI * std::cos(PI * p[0]) * std::sin(PI * p[1]);
    r[1] = PI * std::cos(PI * p[1]) * std::sin(PI * p[0]);
    return r;
  }

  virtual void
  hessian_list(const std::vector<Point<dim>> &        points,
               std::vector<SymmetricTensor<2, dim>> & hessians,
               const unsigned int /*component*/ = 0) const override
  {
    for(unsigned i = 0; i < points.size(); ++i)
    {
      const double x = points[i][0];
      const double y = points[i][1];

      hessians[i][0][0] = -PI * PI * std::sin(PI * x) * std::sin(PI * y);
      hessians[i][0][1] = PI * PI * std::cos(PI * x) * std::cos(PI * y);
      hessians[i][1][1] = -PI * PI * std::sin(PI * x) * std::sin(PI * y);
    }
  }

  double
  bilaplacian(const Point<dim> & p, const unsigned int /*component*/ = 0) const
  {
    const auto & x = p[0];
    const auto & y = p[1];
    return 4 * std::pow(PI, 4.0) * std::sin(PI * x) * std::sin(PI * y);
  }
};



template<int dim>
using Load = ManufacturedLoad<dim, Solution<dim>>;
} // namespace ZeroBoundary



namespace GaussianBells
{
template<int dim>
using Solution = Laplace::Solution<dim>;



template<int dim>
using Load = ManufacturedLoad<dim, Solution<dim>>;
} // namespace GaussianBells

} // namespace Biharmonic

/**
 *
 * PDE data and reference solutions:
 *
 * 1) EquationData
 *
 */
namespace Stokes
{
struct EquationData
{
  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Equation Data:", "");
    oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
    oss << Util::parameter_to_fstring("Assemble pressure mass matrix?:",
                                      assemble_pressure_mass_matrix);
    return oss.str();
  }

  std::set<types::boundary_id> dirichlet_boundary_ids        = {0};
  double                       ip_factor                     = 1.;
  bool                         assemble_pressure_mass_matrix = false;
};



namespace DivergenceFree
{
// Note that the first dim components are the velocity components
// and the last is the pressure.
template<int dim>
class SolutionVelocity : public Function<dim>
{
public:
  SolutionVelocity() : Function<dim>(dim)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;
};

template<>
double
SolutionVelocity<2>::value(const Point<2> & p, const unsigned int component) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  if(component == 0)
    return sin(PI * x);
  if(component == 1)
    return -PI * y * cos(PI * x);

  AssertThrow(false, ExcMessage("Invalid component."));
  return 0;
}

template<>
double
SolutionVelocity<3>::value(const Point<3> & p, const unsigned int component) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);
  const double z = p(2);

  if(component == 0)
    return 2.0 * sin(PI * x);
  if(component == 1)
    return -PI * y * cos(PI * x);
  if(component == 2)
    return -PI * z * cos(PI * x);

  AssertThrow(false, ExcMessage("Invalid component."));
  return 0;
}

template<>
Tensor<1, 2>
SolutionVelocity<2>::gradient(const Point<2> & p, const unsigned int component) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  Tensor<1, 2> return_value;
  if(component == 0)
  {
    return_value[0] = PI * cos(PI * x);
    return_value[1] = 0.0;
  }
  else if(component == 1)
  {
    return_value[0] = y * PI * PI * sin(PI * x);
    return_value[1] = -PI * cos(PI * x);
  }
  else
    AssertThrow(false, ExcMessage("Invalid component."));

  return return_value;
}

template<>
Tensor<1, 3>
SolutionVelocity<3>::gradient(const Point<3> & p, const unsigned int component) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);
  const double z = p(2);

  Tensor<1, 3> return_value;
  if(component == 0)
  {
    return_value[0] = 2 * PI * cos(PI * x);
    return_value[1] = 0.0;
    return_value[2] = 0.0;
  }
  else if(component == 1)
  {
    return_value[0] = y * PI * PI * sin(PI * x);
    return_value[1] = -PI * cos(PI * x);
    return_value[2] = 0.0;
  }
  else if(component == 2)
  {
    return_value[0] = z * PI * PI * sin(PI * x);
    return_value[1] = 0.0;
    return_value[2] = -PI * cos(PI * x);
  }
  else
    AssertThrow(false, ExcMessage("Invalid component."));

  return return_value;
}



template<int dim>
class SolutionPressure : public Function<dim>
{
public:
  SolutionPressure() : Function<dim>(1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;
};

template<>
double
SolutionPressure<2>::value(const Point<2> & p, const unsigned int) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  return sin(PI * x) * cos(PI * y);
}

template<>
double
SolutionPressure<3>::value(const Point<3> & p, const unsigned int) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);
  const double z = p(2);

  return sin(PI * x) * cos(PI * y) * sin(PI * z);
}

template<>
Tensor<1, 2>
SolutionPressure<2>::gradient(const Point<2> & p, const unsigned int) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  Tensor<1, 2> return_value;
  {
    return_value[0] = PI * cos(PI * x) * cos(PI * y);
    return_value[1] = -PI * sin(PI * x) * sin(PI * y);
  }

  return return_value;
}

template<>
Tensor<1, 3>
SolutionPressure<3>::gradient(const Point<3> & p, const unsigned int) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);
  const double z = p(2);

  Tensor<1, 3> return_value;
  {
    return_value[0] = PI * cos(PI * x) * cos(PI * y) * sin(PI * z);
    return_value[1] = -PI * sin(PI * x) * sin(PI * y) * sin(PI * z);
    return_value[2] = PI * sin(PI * x) * cos(PI * y) * cos(PI * z);
  }

  return return_value;
}



template<int dim>
class Solution : public Function<dim>
{
public:
  Solution() : Function<dim>(dim + 1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    if(component < dim)
      return solution_velocity.value(p, component);
    else if(component == dim)
      return solution_pressure.value(p);

    AssertThrow(false, ExcMessage("Invalid component."));
    return 0.;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override
  {
    if(component < dim)
      return solution_velocity.gradient(p, component);
    else if(component == dim)
      return solution_pressure.gradient(p);

    AssertThrow(false, ExcMessage("Invalid component."));
    return Tensor<1, dim>{};
  }

private:
  SolutionVelocity<dim> solution_velocity;
  SolutionPressure<dim> solution_pressure;
};



template<int dim>
class Load : public Function<dim>
{
public:
  Load() : Function<dim>(dim + 1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;
};

template<>
double
Load<2>::value(const Point<2> & p, const unsigned int component) const
{
  Assert(component <= 2, ExcIndexRange(component, 0, 2 + 1));

  using numbers::PI;
  double x = p(0);
  double y = p(1);
  if(component == 0)
    return PI * PI * sin(PI * x) + PI * cos(PI * x) * cos(PI * y);
  if(component == 1)
    return -PI * PI * PI * y * cos(PI * x) - PI * sin(PI * y) * sin(PI * x);
  if(component == 2)
    return 0;

  return 0;
}

template<>
double
Load<3>::value(const Point<3> & p, const unsigned int component) const
{
  Assert(component <= 3, ExcIndexRange(component, 0, 3 + 1));

  using numbers::PI;
  double x = p(0);
  double y = p(1);
  double z = p(2);
  if(component == 0)
    return 2 * PI * PI * sin(PI * x) + PI * cos(PI * x) * cos(PI * y) * sin(PI * z);
  if(component == 1)
    return -PI * PI * PI * y * cos(PI * x) + PI * (-1) * sin(PI * y) * sin(PI * x) * sin(PI * z);
  if(component == 2)
    return -PI * PI * PI * z * cos(PI * x) + PI * cos(PI * z) * sin(PI * x) * cos(PI * y);
  if(component == 3)
    return 0;

  return 0;
}

} // namespace DivergenceFree

} // namespace Stokes

#endif /* EQUATION_DATA_H_ */
