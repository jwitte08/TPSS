/*
 * equation_data.h
 *
 *  Created on: Jul 08, 2019
 *      Author: witte
 */

#ifndef EQUATION_DATA_H_
#define EQUATION_DATA_H_

#include <deal.II/base/function_lib.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_function.h>

#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"

#include "utilities.h"

using namespace dealii;



namespace Common
{
/**
 * This struct defines the parameters used in PolyAtCube.
 */
template<int dim>
struct PolyAtCubeBase
{
  static const std::vector<double> polynomial_coefficients;
};

template<>
const std::vector<double> PolyAtCubeBase<2>::polynomial_coefficients = {{0., 0., 1., -2., 1.}};

template<>
const std::vector<double> PolyAtCubeBase<3>::polynomial_coefficients = {{0., 0., 1., -2., 1.}};



/**
 * Given the univariate polynomial (@p poly)
 *
 *    p(x) = (x-1)^2 * x^2
 *
 * this class represents the polynomial
 *
 *    PHI(x,y) = p(x) * p(y)
 *
 * in two dimensions and the polynomial
 *
 *    PHI(x,y) = p(x) * p(y) * p(z)
 *
 * in three dimensions. The roots of p(x) are chosen such that the value as well
 * as the gradient of PHI(x,y) are zero at the boundary of the unit cube
 * [0,1]^2.
 */
template<int dim>
class PolyAtCube : public Function<dim>, protected PolyAtCubeBase<dim>
{
public:
  PolyAtCube() : Function<dim>(1), poly(PolyAtCubeBase<dim>::polynomial_coefficients)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int /*component*/ = 0) const override;

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int /*component*/ = 0) const override;

  virtual double
  laplacian(const dealii::Point<dim> & p, const unsigned int /*component*/ = 0) const override final
  {
    const double lapl = trace(hessian(p));
    return lapl;
  }

  double
  bilaplacian(const Point<dim> & p, const unsigned int /*component*/ = 0) const;

private:
  Polynomials::Polynomial<double> poly;
};


template<>
double
PolyAtCube<2>::value(const Point<2> & p, const unsigned int /*component*/) const
{
  const auto x = p[0];
  const auto y = p[1];

  std::vector<double> values_x(1U), values_y(1U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  const auto poly_x = values_x[0];
  const auto poly_y = values_y[0];

  return poly_x * poly_y;
}


template<>
double
PolyAtCube<3>::value(const Point<3> & p, const unsigned int /*component*/) const
{
  const auto x = p[0];
  const auto y = p[1];
  const auto z = p[2];

  std::vector<double> values_x(1U), values_y(1U), values_z(1U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  poly.value(z, values_z);
  const auto poly_x = values_x[0];
  const auto poly_y = values_y[0];
  const auto poly_z = values_z[0];

  return poly_x * poly_y * poly_z;
}


template<>
Tensor<1, 2>
PolyAtCube<2>::gradient(const Point<2> & p, const unsigned int /*component*/) const
{
  const auto x = p[0];
  const auto y = p[1];

  std::vector<double> values_x(2U), values_y(2U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  const auto poly_x  = values_x[0]; // p(x)
  const auto Dpoly_x = values_x[1]; // p'(x)
  const auto poly_y  = values_y[0]; // p(y)
  const auto Dpoly_y = values_y[1]; // p'(y)

  Tensor<1, 2> grad;
  grad[0] = Dpoly_x * poly_y;
  grad[1] = poly_x * Dpoly_y;

  return grad;
}


template<>
Tensor<1, 3>
PolyAtCube<3>::gradient(const Point<3> & p, const unsigned int /*component*/) const
{
  const auto x = p[0];
  const auto y = p[1];
  const auto z = p[2];

  std::vector<double> values_x(2U), values_y(2U), values_z(2U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  poly.value(z, values_z);
  const auto poly_x  = values_x[0]; // p(x)
  const auto Dpoly_x = values_x[1]; // p'(x)
  const auto poly_y  = values_y[0]; // p(y)
  const auto Dpoly_y = values_y[1]; // p'(y)
  const auto poly_z  = values_z[0]; // p(z)
  const auto Dpoly_z = values_z[1]; // p'(z)

  Tensor<1, 3> grad;
  grad[0] = Dpoly_x * poly_y * poly_z;
  grad[1] = poly_x * Dpoly_y * poly_z;
  grad[2] = poly_x * poly_y * Dpoly_z;

  return grad;
}


template<>
SymmetricTensor<2, 2>
PolyAtCube<2>::hessian(const Point<2> & p, const unsigned int /*component*/) const
{
  const auto x = p[0];
  const auto y = p[1];

  std::vector<double> values_x(3U), values_y(3U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  const auto poly_x   = values_x[0];
  const auto Dpoly_x  = values_x[1];
  const auto D2poly_x = values_x[2];
  const auto poly_y   = values_y[0];
  const auto Dpoly_y  = values_y[1];
  const auto D2poly_y = values_y[2];

  SymmetricTensor<2, 2> hess;
  hess[0][0] = D2poly_x * poly_y;
  hess[0][1] = Dpoly_x * Dpoly_y;
  hess[1][1] = poly_x * D2poly_y;

  return hess;
}


template<>
SymmetricTensor<2, 3>
PolyAtCube<3>::hessian(const Point<3> & p, const unsigned int /*component*/) const
{
  const auto x = p[0];
  const auto y = p[1];
  const auto z = p[2];

  std::vector<double> values_x(3U), values_y(3U), values_z(3U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  poly.value(z, values_z);
  const auto poly_x   = values_x[0];
  const auto Dpoly_x  = values_x[1];
  const auto D2poly_x = values_x[2];
  const auto poly_y   = values_y[0];
  const auto Dpoly_y  = values_y[1];
  const auto D2poly_y = values_y[2];
  const auto poly_z   = values_z[0];
  const auto Dpoly_z  = values_z[1];
  const auto D2poly_z = values_z[2];

  SymmetricTensor<2, 3> hess;
  hess[0][0] = D2poly_x * poly_y * poly_z;
  hess[0][1] = Dpoly_x * Dpoly_y * poly_z;
  hess[0][2] = Dpoly_x * poly_y * Dpoly_z;
  hess[1][1] = poly_x * D2poly_y * poly_z;
  hess[1][2] = poly_x * Dpoly_y * Dpoly_z;
  hess[2][2] = poly_x * poly_y * D2poly_z;

  return hess;
}


template<>
double
PolyAtCube<2>::bilaplacian(const Point<2> & p, const unsigned int /*component*/) const
{
  const auto & x = p[0];
  const auto & y = p[1];

  std::vector<double> values_x(5U), values_y(5U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  const auto poly_x   = values_x[0];
  const auto D2poly_x = values_x[2];
  const auto D4poly_x = values_x[4];
  const auto poly_y   = values_y[0];
  const auto D2poly_y = values_y[2];
  const auto D4poly_y = values_y[4];

  double bilapl = 0.;
  bilapl        = D4poly_x * poly_y + 2. * D2poly_x * D2poly_y + poly_x * D4poly_y;

  return bilapl;
}


template<>
double
PolyAtCube<3>::bilaplacian(const Point<3> & p, const unsigned int /*component*/) const
{
  const auto & x = p[0];
  const auto & y = p[1];
  const auto & z = p[2];

  std::vector<double> values_x(5U), values_y(5U), values_z(5U);
  poly.value(x, values_x);
  poly.value(y, values_y);
  poly.value(z, values_z);
  const auto poly_x   = values_x[0];
  const auto D2poly_x = values_x[2];
  const auto D4poly_x = values_x[4];
  const auto poly_y   = values_y[0];
  const auto D2poly_y = values_y[2];
  const auto D4poly_y = values_y[4];
  const auto poly_z   = values_z[0];
  const auto D2poly_z = values_z[2];
  const auto D4poly_z = values_z[4];

  double bilapl = 0.;
  bilapl        = D4poly_x * poly_y * poly_z;
  bilapl += 2. * D2poly_x * D2poly_y * poly_z;
  bilapl += 2. * D2poly_x * poly_y * D2poly_z;
  bilapl += poly_x * D4poly_y * poly_z;
  bilapl += 2. * poly_x * D2poly_y * D2poly_z;
  bilapl += poly_x * poly_y * D4poly_z;

  return bilapl;
}

} // namespace Common

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
  enum class Variant
  {
    DirichletHom,     // 0
    DirichletBell,    // 1
    DirichletHomPoly, // 2
  };
  static constexpr unsigned int n_variants = 3;

  static std::string
  str_equation_variant(const Variant variant);

  std::string
  str_equation_variant() const;

  static std::string
  sstr_equation_variant(const Variant variant);

  std::string
  sstr_equation_variant() const;

  std::string
  to_string() const;

  Variant                      variant                = Variant::DirichletBell;
  std::set<types::boundary_id> dirichlet_boundary_ids = {0};
  std::set<types::boundary_id> neumann_boundary_ids   = {};
  double                       ip_factor              = 1.;
};


std::string
EquationData::str_equation_variant(const Variant variant)
{
  std::string str[n_variants] = {"Dirichlet (homogeneous)",
                                 "Dirichlet (Gaussian bells)",
                                 "Dirichlet (polynomial, homogeneous)"};

  return str[static_cast<int>(variant)];
}


std::string
EquationData::str_equation_variant() const
{
  return str_equation_variant(variant);
}


std::string
EquationData::sstr_equation_variant(const Variant variant)
{
  std::string str[n_variants] = {"hom", "bell", "hompoly"};
  return str[static_cast<int>(variant)];
}


std::string
EquationData::sstr_equation_variant() const
{
  return sstr_equation_variant(variant);
}


std::string
EquationData::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Equation Data:", str_equation_variant(variant));
  oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
  oss << Util::parameter_to_fstring("Dirichlet boundary ids:",
                                    set_to_string(dirichlet_boundary_ids));
  oss << Util::parameter_to_fstring("Neumann boundary ids:", set_to_string(neumann_boundary_ids));
  return oss.str();
}



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



namespace GaussianBells
{
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
    return 1. / std::pow(std::sqrt(2 * PI) * width, dim);
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
} // namespace GaussianBells



namespace Homogeneous
{
struct SolutionBase
{
  static constexpr double a = 1.;
};



/**
 * This struct implements the reference solution
 *
 *   PHI(x,y) = sin(pi * x^2) (exp(y*(y-1)^2) - 1)
 *
 * in two dimensions and
 *
 *   PHI(x,y,z) = TODO
 *
 * in three dimensions. PHI is designed to have fulfill zero-Dirichlet
 * conditions at the boundary of the unit hypercube [0,1]^dim.
 */
template<int dim>
class Solution : public Function<dim>, private SolutionBase
{
};

template<>
class Solution<2> : public Function<2>, private SolutionBase
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
class Solution<3> : public Function<3>, private SolutionBase
{
private:
  static constexpr int dim = 3;
  using value_type         = Point<dim>::value_type;
  const Solution<2> func_2d;

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

namespace Poly
{
template<int dim>
using Solution = ::Common::PolyAtCube<dim>;
}

} // namespace Homogeneous



/**
 * Helper struct defines a load function returning random values.
 */
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
enum LocalSolverVariant
{
  Exact,
  Bilaplacian,
  KSVD
};



struct EquationData
{
  enum class Variant
  {
    ClampedHom,                    // 0
    ClampedBell,                   // 1
    ClampedStreamNoSlip,           // 2
    ClampedStreamPoiseuilleNoSlip, // 3
    ClampedStreamNoSlipNormal,     // 4
    ClampedStreamPoiseuilleInhom,  // 5
    ClampedHomPoly                 // 6
  };
  static constexpr unsigned int n_variants = 7;

  static std::string
  str_equation_variant(const Variant variant);

  static std::string
  sstr_equation_variant(const Variant variant);

  std::string
  sstr_equation_variant() const;

  static std::string
  str_local_solver(const LocalSolverVariant variant);

  std::string
  str_local_solver() const;

  std::string
  sstr_local_solver() const;

  std::string
  to_string() const;

  bool
  is_stream_function() const;

  Variant                      variant                         = Variant::ClampedHom;
  std::set<types::boundary_id> dirichlet_boundary_ids          = {0};
  double                       ip_factor                       = 1.;
  LocalSolverVariant           local_solver_variant            = LocalSolverVariant::Exact;
  std::set<unsigned int>       ksvd_tensor_indices             = {0U};
  bool                         force_positive_definite_inverse = false;
  double                       addition_to_min_eigenvalue      = 0.01;
  std::size_t                  n_lanczos_iterations            = static_cast<std::size_t>(-1);
};


std::string
EquationData::str_equation_variant(const Variant variant)
{
  std::string str[n_variants] = {"clamped (homogeneous)",
                                 "clamped (Gaussian bells)",
                                 "clamped (stream function - no-slip)",
                                 "clamped (stream function - no-slip Poiseuille)",
                                 "clamped (stream function - no-slip-normal)",
                                 "clamped (stream function - inhom. Poiseuille)",
                                 "clamped (polynomial, homogeneous)"};
  return str[static_cast<int>(variant)];
}


std::string
EquationData::sstr_equation_variant(const Variant variant)
{
  std::string str[n_variants] = {"clamped_hom",
                                 "clamped_bell",
                                 "clamped_noslip",
                                 "clamped_noslip_poiseuille",
                                 "clamped_noslipnormal",
                                 "clamped_inhom_poiseuille",
                                 "clamped_hompoly"};
  return str[static_cast<int>(variant)];
}


std::string
EquationData::sstr_equation_variant() const
{
  return sstr_equation_variant(variant);
}


std::string
EquationData::str_local_solver(const LocalSolverVariant variant)
{
  const std::string str_variant[] = {"Exact",
                                     "Bilaplacian (no mixed derivatives)",
                                     "Kronecker SVD"};
  return str_variant[(int)variant];
}


std::string
EquationData::str_local_solver() const
{
  return str_local_solver(local_solver_variant);
}


std::string
EquationData::sstr_local_solver() const
{
  const std::string str_variant[] = {"exact", "bilapl", "ksvd"};
  return str_variant[(int)local_solver_variant];
}


std::string
EquationData::to_string() const
{
  std::ostringstream oss;
  oss << Util::parameter_to_fstring("Equation Data:", str_equation_variant(variant));
  oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
  oss << Util::parameter_to_fstring("Local solver:", str_local_solver(local_solver_variant));
  if(local_solver_variant == LocalSolverVariant::KSVD)
  {
    oss << Util::parameter_to_fstring("Selected KSVD tensors:", set_to_string(ksvd_tensor_indices));
    oss << Util::parameter_to_fstring("Number of Lanczos iterations:", n_lanczos_iterations);
    if(force_positive_definite_inverse)
      oss << Util::parameter_to_fstring("Force positive definite inverses:",
                                        force_positive_definite_inverse);
    if(ksvd_tensor_indices.size() == 2U)
      oss << Util::parameter_to_fstring("Addition to min. eigenvalue:", addition_to_min_eigenvalue);
  }
  oss << Util::parameter_to_fstring("Stream function formulation (Stokes):", is_stream_function());
  return oss.str();
}


bool
EquationData::is_stream_function() const
{
  return variant == EquationData::Variant::ClampedStreamNoSlip ||
         variant == EquationData::Variant::ClampedStreamPoiseuilleNoSlip ||
         variant == EquationData::Variant::ClampedStreamNoSlipNormal ||
         variant == EquationData::Variant::ClampedStreamPoiseuilleInhom;
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



namespace Clamped
{
namespace Homogeneous
{
template<int dim>
class Solution : public Function<dim>
{
  static_assert(dim == 2, "Only dim==2 is implemented.");
  static constexpr auto PI = numbers::PI;

public:
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



namespace Poly
{
template<int dim>
using Solution = ::Common::PolyAtCube<dim>;



template<int dim>
using Load = ManufacturedLoad<dim, Solution<dim>>;
} // namespace Poly

} // namespace Homogeneous



namespace GaussianBells
{
template<int dim>
using Solution = Laplace::GaussianBells::Solution<dim>;



template<int dim>
using Load = ManufacturedLoad<dim, Solution<dim>>;
} // namespace GaussianBells



namespace NoSlip
{
template<int dim>
struct SolutionBase
{
  static const std::vector<double> polynomial_coefficients;
};

template<>
const std::vector<double> SolutionBase<2>::polynomial_coefficients = {{0., 0., 1., -2., 1.}};



template<int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  Solution() : Function<dim>(1), poly(SolutionBase<dim>::polynomial_coefficients)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto x = p[0];
    const auto y = p[1];

    std::vector<double> values_x(1U), values_y(1U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x = values_x[0];
    const auto poly_y = values_y[0];

    return poly_x * poly_y;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto x = p[0];
    const auto y = p[1];

    std::vector<double> values_x(2U), values_y(2U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x  = values_x[0];
    const auto Dpoly_x = values_x[1];
    const auto poly_y  = values_y[0];
    const auto Dpoly_y = values_y[1];

    Tensor<1, dim> grad;
    grad[0] = Dpoly_x * poly_y;
    grad[1] = poly_x * Dpoly_y;

    return grad;
  }

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto x = p[0];
    const auto y = p[1];

    std::vector<double> values_x(3U), values_y(3U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x   = values_x[0];
    const auto Dpoly_x  = values_x[1];
    const auto D2poly_x = values_x[2];
    const auto poly_y   = values_y[0];
    const auto Dpoly_y  = values_y[1];
    const auto D2poly_y = values_y[2];

    SymmetricTensor<2, dim> hess;
    hess[0][0] = D2poly_x * poly_y;
    hess[0][1] = Dpoly_x * Dpoly_y;
    hess[1][1] = poly_x * D2poly_y;

    return hess;
  }

  double
  bilaplacian(const Point<dim> & p, const unsigned int /*component*/ = 0) const
  {
    const auto & x = p[0];
    const auto & y = p[1];

    std::vector<double> values_x(5U), values_y(5U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x   = values_x[0];
    const auto D2poly_x = values_x[2];
    const auto D4poly_x = values_x[4];
    const auto poly_y   = values_y[0];
    const auto D2poly_y = values_y[2];
    const auto D4poly_y = values_y[4];

    double bilapl = 0.;
    bilapl        = D4poly_x * poly_y + 2. * D2poly_x * D2poly_y + poly_x * D4poly_y;

    return bilapl;
  }

private:
  Polynomials::Polynomial<double> poly;
};



template<int dim>
using Load = ManufacturedLoad<dim, Solution<dim>>;
} // namespace NoSlip



namespace Poiseuille
{
namespace NoSlip
{
template<int dim>
using Solution = Functions::ZeroFunction<dim>;
}



namespace Inhom
{
template<int dim>
class Solution : public Function<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");
  static constexpr auto PI = numbers::PI;

public:
  virtual double
  value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto y = p[1];

    return y * y * y / 3. - y * y / 2.;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto y = p[1];

    Tensor<1, dim> grad;
    grad[0] = 0.;
    grad[1] = y * (y - 1.);

    return grad;
  }

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto y = p[1];

    SymmetricTensor<2, dim> hess;
    hess[0][0] = 0.;
    hess[0][1] = 0.;
    hess[1][1] = y - 1. + y;

    return hess;
  }

  double
  bilaplacian(const Point<dim> & p, const unsigned int /*component*/ = 0) const
  {
    return 0.;
  }
};

} // namespace Inhom

} // namespace Poiseuille



namespace NoSlipNormal
{
template<int dim>
class Solution : public Function<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");
  static constexpr auto PI = numbers::PI;

public:
  virtual double
  value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto x = p[0];
    const auto y = p[1];

    return sin(PI * x) * cos(PI * y);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto x = p[0];
    const auto y = p[1];

    Tensor<1, dim> grad;
    grad[0] = PI * cos(PI * x) * cos(PI * y);
    grad[1] = -PI * sin(PI * x) * sin(PI * y);

    return grad;
  }

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
  {
    const auto x = p[0];
    const auto y = p[1];

    SymmetricTensor<2, dim> hess;
    hess[0][0] = -PI * PI * sin(PI * x) * cos(PI * y);
    hess[0][1] = -PI * PI * cos(PI * x) * sin(PI * y);
    hess[1][1] = -PI * PI * sin(PI * x) * cos(PI * y);

    return hess;
  }

  double
  bilaplacian(const Point<dim> & p, const unsigned int /*component*/ = 0) const
  {
    const auto & x = p[0];
    const auto & y = p[1];

    constexpr auto PI4    = PI * PI * PI * PI;
    double         bilapl = 0.;
    /// \partial_xxxx
    bilapl += PI4 * sin(PI * x) * cos(PI * y);
    /// 2 \partial_xxyy
    bilapl += 2. * PI4 * sin(PI * x) * cos(PI * y);
    /// \partial_yyyy
    bilapl += PI4 * sin(PI * x) * cos(PI * y);

    return bilapl;
  }
};
} // namespace NoSlipNormal

} // namespace Clamped

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
enum class LocalAssembly
{
  Tensor,
  Cut
};

enum class LocalSolver
{
  Exact,
  Vdiag
};

enum class Method
{
  TaylorHood,
  /*only for testing*/ TaylorHoodDGQ,
  Qkplus2_DGPk,
  DGQkplus2_DGPk,
  RaviartThomas
};



struct EquationData
{
  enum class Variant
  {
    DivFree,                 // 0
    DivFreeNoSlipNormal,     // 1
    DivFreeBell,             // 2
    DivFreePoiseuilleNoSlip, // 3
    DivFreeNoSlip,           // 4
    DivFreePoiseuilleInhom   // 5
  };
  static constexpr unsigned int n_variants = 6;

  static std::string
  str_equation_variant(const Variant variant)
  {
    std::string str[n_variants] = {"divergence-free",
                                   "divergence-free (no-slip-normal)",
                                   "divergence-free (Gaussian bells)",
                                   "divergence-free (no-slip Poiseuille)",
                                   "divergence-free (no-slip)",
                                   "divergence-free (inhom. Poiseuille)"};
    return str[static_cast<int>(variant)];
  }

  static std::string
  sstr_equation_variant(const Variant variant);

  std::string
  sstr_equation_variant() const;

  static std::string
  str_local_solver(const LocalSolver variant)
  {
    std::string str[] = {"exact", "velocity (block diagonal)"};
    return str[static_cast<int>(variant)];
  }

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("Equation Data:", str_equation_variant(variant));
    oss << Util::parameter_to_fstring("IP pre-factor:", ip_factor);
    oss << Util::parameter_to_fstring("Assemble pressure mass matrix?:",
                                      assemble_pressure_mass_matrix);
    oss << Util::parameter_to_fstring("Enforce the mean value constraint:",
                                      force_mean_value_constraint);
    oss << Util::parameter_to_fstring("Use Cuthill-McKee:", use_cuthill_mckee);
    if(local_kernel_size != numbers::invalid_unsigned_int)
      oss << Util::parameter_to_fstring("Kernel size per local solver:", local_kernel_size);
    oss << Util::parameter_to_fstring("Local Solver:", str_local_solver(local_solver));
    return oss.str();
  }

  Variant                      variant                         = Variant::DivFree;
  std::set<types::boundary_id> dirichlet_boundary_ids_velocity = {0};
  std::set<types::boundary_id> dirichlet_boundary_ids_pressure = {};
  double                       ip_factor                       = 1.;
  bool                         assemble_pressure_mass_matrix   = false;
  bool                         force_mean_value_constraint     = false;
  bool                         use_cuthill_mckee               = false;
  unsigned int                 local_kernel_size               = numbers::invalid_unsigned_int;
  double                       local_kernel_threshold          = 0.;
  LocalSolver                  local_solver                    = LocalSolver::Exact;
};


std::string
EquationData::sstr_equation_variant(const Variant variant)
{
  std::string str[] = {
    "tba", "noslipnormal", "bell", "noslip_poiseuille", "noslip", "inhom_poiseuille"};
  return str[static_cast<int>(variant)];
}


std::string
EquationData::sstr_equation_variant() const
{
  return sstr_equation_variant(variant);
}



template<int dim>
class FunctionExtractor : public Function<dim>
{
public:
  /**
   * Extracting the vector components c = start,...,end-1 from function @p
   * function_in, which is determined by the half-open range @p range = [start, end).
   */
  FunctionExtractor(const Function<dim> *                       function_in,
                    const std::pair<unsigned int, unsigned int> range)
    : Function<dim>(range.second - range.first), function(function_in), shift(range.first)
  {
    Assert(range.first <= range.second, ExcMessage("Invalid range."));
    Assert(function_in, ExcMessage("function_in is null"));
    AssertIndexRange(range.first, function_in->n_components);
    AssertIndexRange(range.second, function_in->n_components + 1);
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    return function->value(p, shift + component);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override
  {
    return function->gradient(p, shift + component);
  }

  const Function<dim> * function;
  const unsigned int    shift;
};



template<int dim, typename VelocityFunction, typename PressureFunction>
class FunctionMerge : public Function<dim>
{
public:
  FunctionMerge() : Function<dim>(dim + 1)
  {
    AssertDimension(solution_velocity.n_components, dim);
    AssertDimension(solution_pressure.n_components, 1);
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

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override
  {
    if(component < dim)
      return solution_velocity.hessian(p, component);
    else if(component == dim)
      return solution_pressure.hessian(p, component);

    AssertThrow(false, ExcMessage("Invalid component."));
    return SymmetricTensor<2, dim>{};
  }

private:
  VelocityFunction solution_velocity;
  PressureFunction solution_pressure;
};



template<int dim>
class ManufacturedLoad : public Function<dim>
{
public:
  ManufacturedLoad(const std::shared_ptr<const Function<dim>> solution_function_in)
    : Function<dim>(dim + 1), solution_function(solution_function_in)
  {
    AssertThrow(solution_function_in->n_components == this->n_components,
                ExcMessage("Mismatching number of components."));
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    constexpr auto pressure_index = dim;
    double         value          = 0.;

    /**
     * The manufactured load associated to the velocity block reads
     *
     *    -2 div E + grad p
     *
     * where E = 1/2 (grad u + [grad u]^T) is the linear strain.  The divergence
     * of matrix field E is defined by
     *
     *    (div E)_i = sum_j d/dx_j E_ij.
     *
     * The symmetric gradient E of vector field u reads
     *
     *    E_ij = 1/2 * (d/dx_i u_j + d/dx_j u_i).
     *
     * Combining both, we have
     *
     *    (div E)_i = sum_j 1/2 * (d/dx_j d/dx_i u_j + d^2/dx_j^2 u_i).
     */
    if(component < pressure_index)
    {
      const auto i      = component;
      double     divE_i = 0.;
      for(auto j = 0U; j < dim; ++j)
      {
        const SymmetricTensor<2, dim> & hessian_of_u_j = (solution_function->hessian(p, j));
        const auto   Dji_of_u_j = hessian_of_u_j({i, j} /*TableIndices<2>(i, j)*/);
        const double Djj_of_u_i = (solution_function->hessian(p, i))({j, j});
        divE_i += 0.5 * (Dji_of_u_j + Djj_of_u_i);
      }
      const auto & gradp_i = solution_function->gradient(p, pressure_index)[i];
      value                = -2. * divE_i + gradp_i;
    }

    /**
     * The manufactured load associated to the pressure block reads
     *
     *    - div u
     *
     * with u being the velocity field. The load vanishes for a divergence-free
     * velocity.
     */
    else if(component == pressure_index)
    {
      double divu = 0.;
      for(auto j = 0U; j < dim; ++j)
        divu += solution_function->gradient(p, j)[j];
      value = -divu;
    }

    else
      AssertThrow(false, ExcMessage("Invalid component."));

    return value;
  }

private:
  std::shared_ptr<const Function<dim>> solution_function;
};



template<int dim>
struct PolynomialPressureBase
{
  static const std::vector<double> polynomial_coefficients;

  static Table<2, double>
  get_exponents();
};

template<>
const std::vector<double> PolynomialPressureBase<2>::polynomial_coefficients = {
  {-0.5, 1.}}; // <--- set mean to zero !

template<int dim>
Table<2, double>
PolynomialPressureBase<dim>::get_exponents()
{
  Table<2, double> exp;
  exp.reinit(polynomial_coefficients.size(), dim);
  exp(0, 0) = 0.; // x^0 y^0 z^0 <--- set mean to zero !
  exp(1, 0) = 1.; // x^1 y^0 z^0
  return exp;
}



template<int dim>
class PolynomialPressure : public Functions::Polynomial<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  PolynomialPressure()
    : Functions::Polynomial<dim>(PolynomialPressureBase<dim>::get_exponents(),
                                 PolynomialPressureBase<dim>::polynomial_coefficients)
  {
  }
};



namespace DivergenceFree
{
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

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;
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
SymmetricTensor<2, dim>
SolutionVelocity<dim>::hessian(const Point<dim> &, const unsigned int) const
{
  AssertThrow(false, ExcMessage("Satisfying interfaces..."));
  return SymmetricTensor<2, dim>{};
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

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;
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
SymmetricTensor<2, dim>
SolutionPressure<dim>::hessian(const Point<dim> &, const unsigned int) const
{
  AssertThrow(false, ExcMessage("No need for this functionality..."));
  return SymmetricTensor<2, dim>{};
}



template<int dim>
using Solution = FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;



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



namespace NoSlipNormal
{
/**
 * This class represents the vector curl of
 *
 *    PHI(x,y) = sin(pi*x) cos(pi*y)
 *
 * in two dimensions, which is by definition divergence free and has
 * homogeneous boundary values on the unit cube [0,1]^2.
 */
template<int dim>
class SolutionVelocity : public Function<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  SolutionVelocity() : Function<dim>(dim)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;
};

template<>
double
SolutionVelocity<2>::value(const Point<2> & p, const unsigned int component) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  if(component == 0)
    return PI * sin(PI * x) * cos(PI * y);
  if(component == 1)
    return -PI * cos(PI * x) * sin(PI * y);

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

  Tensor<1, 2> grad;
  if(component == 0)
  {
    grad[0] = PI * PI * cos(PI * x) * cos(PI * y);
    grad[1] = -PI * PI * sin(PI * x) * sin(PI * y);
  }
  else if(component == 1)
  {
    grad[0] = PI * PI * sin(PI * x) * sin(PI * y);
    grad[1] = -PI * PI * cos(PI * x) * cos(PI * y);
  }
  else
    AssertThrow(false, ExcMessage("Invalid component."));

  return grad;
}

template<>
SymmetricTensor<2, 2>
SolutionVelocity<2>::hessian(const Point<2> & p, const unsigned int component) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  SymmetricTensor<2, 2> hessian;
  if(component == 0)
  {
    hessian[0][0] = hessian[1][1] = -PI * PI * PI * sin(PI * x) * cos(PI * y);
    hessian[0][1]                 = -PI * PI * PI * cos(PI * x) * sin(PI * y);
  }
  else if(component == 1)
  {
    hessian[0][0] = hessian[1][1] = PI * PI * PI * cos(PI * x) * sin(PI * y);
    hessian[0][1]                 = PI * PI * PI * sin(PI * x) * cos(PI * y);
  }
  else
    AssertThrow(false, ExcMessage("Invalid component."));

  return hessian;
}



template<int dim>
class SolutionPressure : public Function<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  SolutionPressure() : Function<dim>(1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;
};

template<>
double
SolutionPressure<2>::value(const Point<2> & p, const unsigned int) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  return cos(2. * PI * x) * cos(2. * PI * y);
}

template<>
Tensor<1, 2>
SolutionPressure<2>::gradient(const Point<2> & p, const unsigned int) const
{
  using numbers::PI;
  const double x = p(0);
  const double y = p(1);

  Tensor<1, 2> grad;
  {
    grad[0] = -2. * PI * sin(2. * PI * x) * cos(2. * PI * y);
    grad[1] = -2. * PI * cos(2. * PI * x) * sin(2. * PI * y);
  }

  return grad;
}

template<>
SymmetricTensor<2, 2>
SolutionPressure<2>::hessian(const Point<2> &, const unsigned int) const
{
  AssertThrow(false, ExcMessage("No need for this functionality..."));
  return SymmetricTensor<2, 2>{};
}



template<int dim>
using Solution = FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;

} // namespace NoSlipNormal



namespace GaussianBell
{
template<int dim>
using SolutionBaseVelocity = Laplace::GaussianBells::SolutionBase<dim>;



/**
 * This class represents the vector curl of
 *
 *    PHI(x,y) = N * \sum_i exp[-||x-x_i||^2 / s^2]
 *
 * in two dimensions, i.e. a linear combination of Gaussian bell curves with
 * center point x_i and normalization factor N = 1 / (2Pi * s^2)^{d/2}. This
 * velocity field is by definition divergence free and has heterogeneous
 * boundary values on the unit cube [0,1]^2.
 */
template<int dim>
class SolutionVelocity : public Function<dim>, protected SolutionBaseVelocity<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  using SolutionBaseVelocity<dim>::width;
  using SolutionBaseVelocity<dim>::n_source_centers;
  using SolutionBaseVelocity<dim>::source_centers;
  static constexpr auto PI = numbers::PI;

  SolutionVelocity() : Function<dim>(dim)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;

private:
  constexpr double
  u0() const
  {
    return 1. / std::pow(std::sqrt(2 * PI) * width, dim);
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
    const dealii::Tensor<1, dim> x_minus_xi = p - source_centers[i];
    return std::exp(v0() * x_minus_xi.norm_square());
  }

  double
  value_impl(const Point<dim> & p, const unsigned int component, const unsigned int i) const;

  Tensor<1, dim>
  gradient_impl(const Point<dim> & p, const unsigned int component, const unsigned int i) const;

  double
  laplacian_impl(const dealii::Point<dim> & p,
                 const unsigned int         component,
                 const unsigned int         i) const;

  SymmetricTensor<2, dim>
  hessian_impl(const Point<dim> & p, const unsigned int component, const unsigned int i) const;
};

template<>
double
SolutionVelocity<2>::value_impl(const Point<2> &   p,
                                const unsigned int comp,
                                const unsigned int i) const
{
  AssertIndexRange(comp, 2);
  const unsigned int         anti_comp  = 1 - comp;
  const dealii::Tensor<1, 2> x_minus_xi = p - source_centers[i];
  constexpr double           sign[]     = {1., -1.};

  return sign[comp] * 2. * u0() * v0() * bell(p, i) * x_minus_xi[anti_comp];
}

template<>
Tensor<1, 2>
SolutionVelocity<2>::gradient_impl(const Point<2> &   p,
                                   const unsigned int comp,
                                   const unsigned int i) const
{
  AssertIndexRange(comp, 2);
  const unsigned int         anti_comp  = 1 - comp;
  const dealii::Tensor<1, 2> x_minus_xi = p - source_centers[i];
  constexpr double           sign[]     = {1., -1.};
  Tensor<1, 2>               grad;

  grad[comp] = sign[comp] * std::pow(2., 2) * bell(p, i) * u0() * v0() * v0() *
               x_minus_xi[anti_comp] * x_minus_xi[comp];

  const double val3 = 1.;
  const double val4 = 2. * v0() * std::pow(x_minus_xi[anti_comp], 2);
  grad[anti_comp]   = sign[comp] * 2. * bell(p, i) * u0() * v0() * (val3 + val4);

  return grad;
}

template<>
SymmetricTensor<2, 2>
SolutionVelocity<2>::hessian_impl(const Point<2> &   p,
                                  const unsigned int comp,
                                  const unsigned int i) const
{
  AssertIndexRange(comp, 2);
  const unsigned int         anti_comp  = 1 - comp;
  const dealii::Tensor<1, 2> x_minus_xi = p - source_centers[i];
  constexpr double           sign[]     = {1., -1.};

  SymmetricTensor<2, 2> hess;
  for(auto d1 = 0U; d1 < 2; ++d1)
    for(auto d2 = d1; d2 < 2; ++d2)
    {
      const double factor = ((d1 == d2 && d1 == anti_comp) ? 3. : 1.);
      const double val1   = factor * std::pow(2., 2) * x_minus_xi[d1 == d2 ? anti_comp : comp];
      const double val2 =
        +std::pow(2., 3) * v0() * x_minus_xi[d1] * x_minus_xi[d2] * x_minus_xi[anti_comp];

      hess[d1][d2] = sign[comp] * bell(p, i) * u0() * v0() * v0() * (val1 + val2);
    }

  return hess;
}

template<>
double
SolutionVelocity<2>::value(const Point<2> & p, const unsigned int component) const
{
  double val = 0;
  for(unsigned int i = 0; i < n_source_centers; ++i)
    val += value_impl(p, component, i);
  return val;
}

template<>
Tensor<1, 2>
SolutionVelocity<2>::gradient(const Point<2> & p, const unsigned int component) const
{
  Tensor<1, 2> grad;
  for(unsigned int i = 0; i < n_source_centers; ++i)
    grad += gradient_impl(p, component, i);
  return grad;
}

template<>
SymmetricTensor<2, 2>
SolutionVelocity<2>::hessian(const Point<2> & p, const unsigned int component) const
{
  SymmetricTensor<2, 2> hess;
  for(unsigned int i = 0; i < n_source_centers; ++i)
    hess += hessian_impl(p, component, i);
  return hess;
}



template<int dim>
using SolutionPressure = NoSlipNormal::SolutionPressure<dim>;



template<int dim>
class SolutionBasePressureAlt
{
protected:
  static const Point<dim> source;
  static const double     width;
};

template<>
const Point<2> SolutionBasePressureAlt<2>::source = Point<2>(0.5, 0.);

template<int dim>
const double SolutionBasePressureAlt<dim>::width = 1.5;



/**
 * This class represents the scalar pressure field
 *
 *    PHI(x,y) = N * exp[-||x-\mu||^2 / s^2] - MEAN
 *
 * in two dimensions, i.e. an normalized Gaussian bell curve around the center
 * point \mu and normalization factor N = 1 / (2Pi * s^2)^{d/2} from which we
 * subtract the hardcoded mean value. To this end, if you modify function
 * parameters keep in mind to modify the mean as well.
 */
template<int dim>
class SolutionPressureAlt : public Function<dim>, protected SolutionBasePressureAlt<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  using SolutionBasePressureAlt<dim>::source;
  using SolutionBasePressureAlt<dim>::width;
  static constexpr auto PI = numbers::PI;

  SolutionPressureAlt() : Function<dim>(1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;

private:
  constexpr double
  u0() const
  {
    return 1. / std::pow(std::sqrt(2 * PI) * width, dim);
  }

  constexpr double
  v0() const
  {
    return -1. / (width * width);
  }

  /**
   * Hardcoded mean. Assumes unit cube [0,1]^2 and source (0.5, 0.0).
   */
  double
  mean() const
  {
    return 1. / 4. * std::erf(1. / (2. * width)) * std::erf(1. / width);
  }

  double
  bell(const Point<dim> & p) const
  {
    const dealii::Tensor<1, dim> x_minus_xi = p - source;
    return std::exp(v0() * x_minus_xi.norm_square());
  }
};

template<>
double
SolutionPressureAlt<2>::value(const Point<2> & p, const unsigned int) const
{
  return u0() * bell(p) - mean();
}

template<>
Tensor<1, 2>
SolutionPressureAlt<2>::gradient(const Point<2> & p, const unsigned int) const
{
  const dealii::Tensor<1, 2> x_minus_xi = p - source;
  return u0() * 2. * v0() * bell(p) * x_minus_xi;
}

template<>
SymmetricTensor<2, 2>
SolutionPressureAlt<2>::hessian(const Point<2> &, const unsigned int) const
{
  AssertThrow(false, ExcMessage("No need for this functionality..."));
  return SymmetricTensor<2, 2>{};
}



template<int dim>
using Solution = FunctionMerge<dim, SolutionVelocity<dim>, NoSlipNormal::SolutionPressure<dim>>;

} // namespace GaussianBell



namespace Poiseuille
{
/**
 * This class represents the scalar pressure field
 *
 *    PHI(x,y) = -Pi x + Pi/2
 *
 * in two dimensions.
 */
template<int dim>
class SolutionPressure : public Function<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  static constexpr auto PI = numbers::PI;

  SolutionPressure() : Function<dim>(1)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override;

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override;
};

template<>
double
SolutionPressure<2>::value(const Point<2> & p, const unsigned int) const
{
  const auto & x = p[0];
  return -PI * x + PI / 2.;
}

template<>
Tensor<1, 2>
SolutionPressure<2>::gradient(const Point<2> &, const unsigned int) const
{
  const Tensor<1, 2> grad{{-PI, 0}};
  return grad;
}

template<>
SymmetricTensor<2, 2>
SolutionPressure<2>::hessian(const Point<2> &, const unsigned int) const
{
  AssertThrow(false, ExcMessage("No need for this functionality..."));
  return SymmetricTensor<2, 2>{};
}



namespace NoSlip
{
template<int dim>
class SolutionVelocity : public Functions::ZeroFunction<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  SolutionVelocity() : Functions::ZeroFunction<dim>(dim)
  {
  }
};


template<int dim>
using Solution = FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;
} // namespace NoSlip



namespace Inhom
{
template<int dim>
class SolutionVelocity : public Function<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");

public:
  SolutionVelocity() : Function<dim>(dim)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    const double y = p(1);

    if(component == 0)
      return y * (y - 1.);
    if(component == 1)
      return 0.;

    AssertThrow(false, ExcMessage("Invalid component."));
    return 0.;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);

    const double y = p(1);

    Tensor<1, dim> grad;
    if(component == 0)
    {
      grad[0] = 0.;
      grad[1] = (y - 1.) + y;
    }
    else if(component == 1)
    {
      grad[0] = 0.;
      grad[1] = 0.;
    }
    else
      AssertThrow(false, ExcMessage("Invalid component."));

    return grad;
  }

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> &, const unsigned int component = 0) const override
  {
    SymmetricTensor<2, dim> hess;
    if(component == 0)
    {
      hess[0][0] = 0.;
      hess[0][1] = 0.;
      hess[1][1] = 2.;
    }
    else if(component == 1)
    {
      hess[0][0] = 0.;
      hess[0][1] = 0.;
      hess[1][1] = 0.;
    }

    else
      AssertThrow(false, ExcMessage("Invalid component."));

    return hess;
  }
};



template<int dim>
using Solution = FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;
} // namespace Inhom

} // namespace Poiseuille



namespace NoSlip
{
/**
 * Given the univariate polynomial (@p poly)
 *
 *    p(x) = (x-1)^2 * x^2
 *
 * this class represents the vector curl of
 *
 *    PHI(x,y) = p(x) * p(y)
 *
 * in two dimensions. The roots of p(x) lead to no-slip boundary conditions on
 * the unit cube [0,1]^2. This is the reference solution for the stream function
 * formulation in Kanschat, Sharma '14, thus closely connected to the biharmonic
 * problem of stream functions with clamped boundary conditions.
 */
template<int dim>
class SolutionVelocity : public Function<dim>,
                         protected Biharmonic::Clamped::NoSlip::SolutionBase<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");
  using Biharmonic::Clamped::NoSlip::SolutionBase<dim>::polynomial_coefficients;

public:
  SolutionVelocity() : Function<dim>(dim), poly(polynomial_coefficients)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);

    const auto          x = p[0];
    const auto          y = p[1];
    std::vector<double> values_x(2U), values_y(2U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x  = values_x[0];
    const auto Dpoly_x = values_x[1];
    const auto poly_y  = values_y[0];
    const auto Dpoly_y = values_y[1];

    double val = 0.;
    if(component == 0U)
      val = poly_x * Dpoly_y;
    else if(component == 1U)
      val = -Dpoly_x * poly_y;
    else
      AssertThrow(false, ExcMessage("Invalid component."));
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);

    const auto x = p[0];
    const auto y = p[1];

    std::vector<double> values_x(3U), values_y(3U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x   = values_x[0];
    const auto Dpoly_x  = values_x[1];
    const auto D2poly_x = values_x[2];
    const auto poly_y   = values_y[0];
    const auto Dpoly_y  = values_y[1];
    const auto D2poly_y = values_y[2];

    Tensor<1, dim> grad;
    if(component == 0U)
    {
      grad[0] = Dpoly_x * Dpoly_y;
      grad[1] = poly_x * D2poly_y;
    }
    else if(component == 1U)
    {
      grad[0] = -D2poly_x * poly_y;
      grad[1] = -Dpoly_x * Dpoly_y;
    }
    else
      AssertThrow(false, ExcMessage("Invalid component."));
    return grad;
  }

  virtual SymmetricTensor<2, dim>
  hessian(const Point<dim> & p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);

    const auto x = p[0];
    const auto y = p[1];

    std::vector<double> values_x(4U), values_y(4U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x   = values_x[0];
    const auto Dpoly_x  = values_x[1];
    const auto D2poly_x = values_x[2];
    const auto D3poly_x = values_x[3];
    const auto poly_y   = values_y[0];
    const auto Dpoly_y  = values_y[1];
    const auto D2poly_y = values_y[2];
    const auto D3poly_y = values_y[3];

    SymmetricTensor<2, dim> hess;
    if(component == 0U)
    {
      hess[0][0] = D2poly_x * Dpoly_y;
      hess[0][1] = Dpoly_x * D2poly_y;
      hess[1][1] = poly_x * D3poly_y;
    }
    else if(component == 1U)
    {
      hess[0][0] = -D3poly_x * poly_y;
      hess[0][1] = -D2poly_x * Dpoly_y;
      hess[1][1] = -Dpoly_x * D2poly_y;
    }
    else
      AssertThrow(false, ExcMessage("Invalid component."));
    return hess;
  }

private:
  Polynomials::Polynomial<double> poly;
};



// /**
//  * Choosing a zero pressure results in a divergence-free manufactured load.
//  */
// template<int dim>
// using SolutionPressure = Functions::ZeroFunction<dim>;

// !!!
template<int dim>
using SolutionPressure = NoSlipNormal::SolutionPressure<dim>;

// !!!
//   template<int dim>
// using SolutionPressure = PolynomialPressure<dim>;

template<int dim>
using Solution = FunctionMerge<dim, SolutionVelocity<dim>, SolutionPressure<dim>>;



/**
 * The explicit definition of the divergence-free load which suits the
 * velocity and pressure reference solutions from above.
 */
template<int dim>
class Load : public Function<dim>, protected Biharmonic::Clamped::NoSlip::SolutionBase<dim>
{
  static_assert(dim == 2, "Implemented for two dimensions.");
  using Biharmonic::Clamped::NoSlip::SolutionBase<dim>::polynomial_coefficients;

public:
  // TODO init @p poly by the coefficients @p polynomial_coefficients
  Load() : Function<dim>(dim + 1), poly(std::vector<double>{{0., 0., 1., -2., 1.}})
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    constexpr auto pressure_index = dim;
    double         value          = 0.;

    const auto x = p[0];
    const auto y = p[1];

    std::vector<double> values_x(4U), values_y(4U);
    poly.value(x, values_x);
    poly.value(y, values_y);
    const auto poly_x   = values_x[0];
    const auto Dpoly_x  = values_x[1];
    const auto D2poly_x = values_x[2];
    const auto D3poly_x = values_x[3];
    const auto poly_y   = values_y[0];
    const auto Dpoly_y  = values_y[1];
    const auto D2poly_y = values_y[2];
    const auto D3poly_y = values_y[3];

    if(component < pressure_index)
    {
      if(component == 0U)
      {
        value = -D2poly_x * Dpoly_y - poly_x * D3poly_y;
      }
      else if(component == 1U)
      {
        value = D3poly_x * poly_y + Dpoly_x * D2poly_y;
      }
      else
        AssertThrow(false, ExcMessage("Invalid component."));
    }

    else if(component == pressure_index)
    {
      value = 0.;
    }

    else
      AssertThrow(false, ExcMessage("Invalid component."));

    return value;
  }

  Polynomials::Polynomial<double> poly;
};

} // namespace NoSlip

} // namespace DivergenceFree

} // namespace Stokes

#endif /* EQUATION_DATA_H_ */
