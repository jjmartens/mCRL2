// Author(s): Jan Friso Groote
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pres/ressolve.h
/// \brief This contains a gauss-elimination like algorithm to solve a res

#ifndef MCRL2_PRES_RESSOLVE_H
#define MCRL2_PRES_RESSOLVE_H

#include "mcrl2/data/real_utilities.h"
#include "mcrl2/pres/builder.h" 

namespace mcrl2 {

namespace pres_system {

/// \brief A builder that substitutes an expression for a propositional variable instantiation in a pres expression. 
///
class substitute_pres_equation_builder: public pres_expression_builder < substitute_pres_equation_builder >
{
protected:
  core::identifier_string m_pres_variable_name;
  pres_expression m_term;

public:
  typedef  pres_expression_builder < substitute_pres_equation_builder > super;
  using super::apply;

  substitute_pres_equation_builder(const propositional_variable& variable, const pres_expression& t)
   : m_pres_variable_name(variable.name()),
     m_term(t)
  {}

  template <class T>
  void apply(T& result, const pres_system::propositional_variable_instantiation& x)
  {
    if (x.name()==m_pres_variable_name)
    {
      result = m_term;
    }
    else
    {
      result = x;
    }
  }

};


namespace detail {

void push_and_inside(pres_expression& result, const pres_expression t1, const pres_expression& t2, const bool conjunctive_normal_form)
{
  pres_expression aux;
  if (is_condsm(t1))
  {
    push_and_inside(aux, atermpp::down_cast<condsm>(t1).arg2(), t2, conjunctive_normal_form);
    push_and_inside(result, atermpp::down_cast<condsm>(t1).arg3(), t2, conjunctive_normal_form);
    optimized_condsm(result, atermpp::down_cast<condsm>(t1).arg1(), aux, result);
  }
  else if (is_condsm(t2))
  {
    push_and_inside(aux, t1, atermpp::down_cast<condsm>(t2).arg2(), conjunctive_normal_form);
    push_and_inside(result, t1, atermpp::down_cast<condsm>(t2).arg3(), conjunctive_normal_form);
    optimized_condsm(result, atermpp::down_cast<condsm>(t2).arg1(), aux, result);
  }
  else if (is_condeq(t1))
  {
    push_and_inside(aux, atermpp::down_cast<condeq>(t1).arg2(), t2, conjunctive_normal_form);
    push_and_inside(result, atermpp::down_cast<condeq>(t1).arg3(), t2, conjunctive_normal_form);
    optimized_condeq(result, atermpp::down_cast<condeq>(t1).arg1(), aux, result);
  }
  else if (is_condeq(t2))
  {
    push_and_inside(aux, t1, atermpp::down_cast<condeq>(t2).arg2(), conjunctive_normal_form);
    push_and_inside(result, t1, atermpp::down_cast<condeq>(t2).arg3(), conjunctive_normal_form);
    optimized_condeq(result, atermpp::down_cast<condeq>(t2).arg1(), aux, result);
  }
  else if (!conjunctive_normal_form && is_or(t1))
  {
    push_and_inside(aux, atermpp::down_cast<or_>(t1).left(), t2, conjunctive_normal_form);
    push_and_inside(result, atermpp::down_cast<or_>(t1).right(), t2, conjunctive_normal_form);
    optimized_or(result, aux, result);
  }
  else if (!conjunctive_normal_form && is_or(t2))
  {
    push_and_inside(aux, t1, atermpp::down_cast<or_>(t2).left(), conjunctive_normal_form);
    push_and_inside(result, t1, atermpp::down_cast<or_>(t2).right(), conjunctive_normal_form);
    optimized_or(result, aux, result);
  }
  else optimized_and(result, t1, t2);
}

void push_or_inside(pres_expression& result, const pres_expression t1, const pres_expression& t2, const bool conjunctive_normal_form)
{ 
  pres_expression aux;
  if (is_condsm(t1))
  { 
    push_or_inside(aux, atermpp::down_cast<condsm>(t1).arg2(), t2, conjunctive_normal_form);
    push_or_inside(result, atermpp::down_cast<condsm>(t1).arg3(), t2, conjunctive_normal_form);
    optimized_condsm(result, atermpp::down_cast<condsm>(t1).arg1(), aux, result);
  }
  else if (is_condsm(t2))
  { 
    push_or_inside(aux, t1, atermpp::down_cast<condsm>(t2).arg2(), conjunctive_normal_form);
    push_or_inside(result, t1, atermpp::down_cast<condsm>(t2).arg3(), conjunctive_normal_form);
    optimized_condsm(result, atermpp::down_cast<condsm>(t2).arg1(), aux, result);
  }
  else if (is_condeq(t1))
  { 
    push_or_inside(aux, atermpp::down_cast<condeq>(t1).arg2(), t2, conjunctive_normal_form);
    push_or_inside(result, atermpp::down_cast<condeq>(t1).arg3(), t2, conjunctive_normal_form);
    optimized_condeq(result, atermpp::down_cast<condeq>(t1).arg1(), aux, result);
  }
  else if (is_condeq(t2))
  { 
    push_or_inside(aux, t1, atermpp::down_cast<condeq>(t2).arg2(), conjunctive_normal_form);
    push_or_inside(result, t1, atermpp::down_cast<condeq>(t2).arg3(), conjunctive_normal_form);
    optimized_condeq(result, atermpp::down_cast<condeq>(t2).arg1(), aux, result);
  }
  else if (conjunctive_normal_form && is_and(t1))
  { 
    push_or_inside(aux, atermpp::down_cast<and_>(t1).left(), t2, conjunctive_normal_form);
    push_or_inside(result, atermpp::down_cast<and_>(t1).right(), t2, conjunctive_normal_form);
    optimized_or(result, aux, result);
  }
  else if (conjunctive_normal_form && is_and(t2))
  { 
    push_or_inside(aux, t1, atermpp::down_cast<and_>(t2).left(), conjunctive_normal_form);
    push_or_inside(result, t1, atermpp::down_cast<and_>(t2).right(), conjunctive_normal_form);
    optimized_or(result, aux, result);
  }
  else optimized_or(result, t1, t2);
} 

void push_plus_inside(pres_expression& result, const pres_expression t1, const pres_expression& t2, const bool conjunctive_normal_form)
{
  pres_expression aux;
  if (is_condsm(t1))
  {
    push_plus_inside(aux, atermpp::down_cast<condsm>(t1).arg2(), t2, conjunctive_normal_form);
    push_plus_inside(result, atermpp::down_cast<condsm>(t1).arg3(), t2, conjunctive_normal_form);
    optimized_condsm(result, atermpp::down_cast<condsm>(t1).arg1(), aux, result);
  }
  else if (is_condsm(t2))
  {
    push_plus_inside(aux, t1, atermpp::down_cast<condsm>(t2).arg2(), conjunctive_normal_form);
    push_plus_inside(result, t1, atermpp::down_cast<condsm>(t2).arg3(), conjunctive_normal_form);
    optimized_condsm(result, atermpp::down_cast<condsm>(t2).arg1(), aux, result);
  }
  else if (is_condeq(t1))
  {
    push_plus_inside(aux, atermpp::down_cast<condeq>(t1).arg2(), t2, conjunctive_normal_form);
    push_plus_inside(result, atermpp::down_cast<condeq>(t1).arg3(), t2, conjunctive_normal_form);
    optimized_condeq(result, atermpp::down_cast<condeq>(t1).arg1(), aux, result);
  }
  else if (is_condeq(t2))
  {
    push_plus_inside(aux, t1, atermpp::down_cast<condeq>(t2).arg2(), conjunctive_normal_form);
    push_plus_inside(result, t1, atermpp::down_cast<condeq>(t2).arg3(), conjunctive_normal_form);
    optimized_condeq(result, atermpp::down_cast<condeq>(t2).arg1(), aux, result);
  }
  else if (conjunctive_normal_form && is_and(t1))  // CNF: first move || upwards.
  {
    push_plus_inside(aux, atermpp::down_cast<and_>(t1).left(), t2, conjunctive_normal_form);
    push_plus_inside(result, atermpp::down_cast<and_>(t1).right(), t2, conjunctive_normal_form);
    optimized_and(result, aux, result);
  }
  else if (conjunctive_normal_form && is_and(t2))
  {
    push_plus_inside(aux, t1, atermpp::down_cast<and_>(t2).left(), conjunctive_normal_form);
    push_plus_inside(result, t1, atermpp::down_cast<and_>(t2).right(), conjunctive_normal_form);
    optimized_and(result, aux, result);
  }
  else if (is_or(t1))
  {
    push_plus_inside(aux, atermpp::down_cast<or_>(t1).left(), t2, conjunctive_normal_form);
    push_plus_inside(result, atermpp::down_cast<or_>(t1).right(), t2, conjunctive_normal_form);
    optimized_or(result, aux, result);
  }
  else if (is_or(t2))
  {
    push_plus_inside(aux, t1, atermpp::down_cast<or_>(t2).left(), conjunctive_normal_form);
    push_plus_inside(result, t1, atermpp::down_cast<or_>(t2).right(), conjunctive_normal_form);
    optimized_or(result, aux, result);
  }
  else if (!conjunctive_normal_form && is_and(t1))  // DNF: first more || upwards. 
  {
    push_plus_inside(aux, atermpp::down_cast<and_>(t1).left(), t2, conjunctive_normal_form);
    push_plus_inside(result, atermpp::down_cast<and_>(t1).right(), t2, conjunctive_normal_form);
    optimized_and(result, aux, result);
  }
  else if (!conjunctive_normal_form && is_and(t2))
  {
    push_plus_inside(aux, t1, atermpp::down_cast<and_>(t2).left(), conjunctive_normal_form);
    push_plus_inside(result, t1, atermpp::down_cast<and_>(t2).right(), conjunctive_normal_form);
    optimized_and(result, aux, result);
  }
  else optimized_plus(result, t1, t2);
}

void push_constant_inside(pres_expression& result, const data::data_expression& constant, const pres_expression& t)
{
  pres_expression aux;
  if (is_true(t))
  {
    result=true_();
  }
  else if (is_false(t))
  {
    result=false_();
  }
  else if (data::is_data_expression(t))
  {
    data::sort_real::make_times(atermpp::reference_cast<data::data_expression>(result),
                                constant,
                                atermpp::down_cast<data::data_expression>(t));
  }
  else if (is_condsm(t))
  { 
    push_constant_inside(aux, constant, atermpp::down_cast<condsm>(t).arg2());
    push_constant_inside(result, constant, atermpp::down_cast<condsm>(t).arg3());
    optimized_condsm(result, atermpp::down_cast<condsm>(t).arg1(), aux, result);
  }
  else if (is_condeq(t))
  {
    push_constant_inside(aux, constant, atermpp::down_cast<condeq>(t).arg2());
    push_constant_inside(result, constant, atermpp::down_cast<condeq>(t).arg3());
    optimized_condeq(result, atermpp::down_cast<condeq>(t).arg1(), aux, result);
  }
  else if (is_and(t))
  {
    push_constant_inside(aux, constant, atermpp::down_cast<and_>(t).left());
    push_constant_inside(result, constant, atermpp::down_cast<and_>(t).right());
    optimized_and(result, aux, result);
  }
  else if (is_or(t))
  {
    pres_expression aux;
    push_constant_inside(aux, constant, atermpp::down_cast<or_>(t).left());
    push_constant_inside(result, constant, atermpp::down_cast<or_>(t).right());
    optimized_or(result, aux, result);
  }
  else if (is_plus(t))
  {
    pres_expression aux;
    push_constant_inside(aux, constant, atermpp::down_cast<plus>(t).left());
    push_constant_inside(result, constant, atermpp::down_cast<plus>(t).right());
    optimized_plus(result, aux, result);
  }
  else if (is_const_multiply(t))
  {
    const const_multiply& tcm = atermpp::down_cast<const_multiply>(t);
    data::data_expression result_term;
    data::sort_real::make_times(result_term, tcm.left(), constant);
    optimized_const_multiply(result, result_term, tcm.right());
  }
  else if (is_const_multiply_alt(t))
  {
    const const_multiply_alt& tcm = atermpp::down_cast<const_multiply_alt>(t);
    data::data_expression result_term;
    data::sort_real::make_times(result_term, tcm.right(), constant);
    optimized_const_multiply(result, result_term, tcm.left());
  }
  else
  {
    optimized_const_multiply(result, constant, t);
  }
}

struct linear_fixed_point_equation
{
  data::data_expression c_j;
  pres_expression f_j;
  bool c_j_term_present=false;    // is true if c_j > 0.
  bool f_j_term_present=false;
  bool eqninf_term_present=false;      // is true if c_j' = 1.

  void update_f_j(const pres_expression& t)
  {
    if (f_j_term_present)
    {
      f_j=plus(f_j, t);
    }
    else
    {
      f_j_term_present=true;
      f_j = t;
    }
  }

  void update_c_j(const data::data_expression& t)
  {
    if (c_j_term_present)
    {
      c_j=data::sort_real::plus(c_j, t);
    }
    else
    {
      c_j_term_present=true;
      c_j = t;
    }
  }
};

// Collect the linear equations for variable v in the current conjunct/disjunct. 
void collect_line(linear_fixed_point_equation& line, const  propositional_variable& v, const pres_expression& t, const bool minimal_fixed_point)
{
  if (is_plus(t))
  {
    collect_line(line, v, atermpp::down_cast<plus>(t).left(), minimal_fixed_point);
    collect_line(line, v, atermpp::down_cast<plus>(t).right(), minimal_fixed_point);
  }
  else if (is_propositional_variable_instantiation(t))
  {
    if (v.name()==atermpp::down_cast<propositional_variable_instantiation>(t).name())
    {
      line.update_c_j(data::sort_real::real_one());  // Add this inclination for v to all inclinations. 
    }
    else
    {
      line.update_f_j(t);
    }
  }
  else if (is_const_multiply(t))
  {
    const const_multiply& tcm = atermpp::down_cast<const_multiply>(t);
    const propositional_variable_instantiation& w = atermpp::down_cast<propositional_variable_instantiation>(tcm.right());
    if (v.name()==w.name())
    {
      line.update_c_j(tcm.left());
    }
    else 
    {
      line.update_f_j(t);
    }
  }
  else if (is_eqninf(t))
  {
    const eqninf& te = atermpp::down_cast<eqninf>(t);
    const propositional_variable_instantiation& w = atermpp::down_cast<propositional_variable_instantiation>(te.operand());
    if (v.name()==w.name())
    {
      line.eqninf_term_present=true;
    }
    else
    {
      line.update_f_j(t);
    }
  }
  else if (data::is_data_expression(t))
  {
    line.update_f_j(t);
  }
  else 
  {
    throw runtime_error("Unexpected term in collect line. Expect plus, constant multiplication or a data expression: " + pp(t) + ". ");
  }
}

void collect_lines(std::vector< linear_fixed_point_equation >& found_lines, 
                   const propositional_variable& v, 
                   const pres_expression& t,
                   const bool minimal_fixed_point)
{
  if (is_and(t))
  {
    and_ ta = atermpp::down_cast<and_>(t);
    collect_lines(found_lines, v, ta.left(), minimal_fixed_point);
    collect_lines(found_lines, v, ta.right(), minimal_fixed_point);
  }
  else if (is_or(t))
  {
    or_ to = atermpp::down_cast<or_>(t);
    collect_lines(found_lines, v, to.left(), minimal_fixed_point);
    collect_lines(found_lines, v, to.right(), minimal_fixed_point);
  }
  else
  {
    found_lines.emplace_back();
    collect_line(found_lines.back(), v, t, minimal_fixed_point);
  }
}

void collect_m_and_split_lines(const std::vector< linear_fixed_point_equation >& lines, 
                               pres_expression& m, 
                               std::vector< linear_fixed_point_equation >& shallow_lines, 
                               std::vector< linear_fixed_point_equation >& steep_lines,
                               std::vector< linear_fixed_point_equation >& flat_lines,
                               const data::rewriter& rewriter,
                               const bool minimal_fixed_point)
{
  bool m_defined=false;
  m = minimal_fixed_point?data::false_():data::true_();
  for(const linear_fixed_point_equation& eq: lines)
  {
    if (eq.c_j_term_present)
    {
      data::data_expression is_shallow=rewriter(data::less(eq.c_j,data::sort_real::real_one()));
      if (data::is_true(is_shallow))
      {
        shallow_lines.push_back(eq);
      }
      else if (data::is_false(is_shallow))
      {
        steep_lines.push_back(eq);
      }
      else
      {
        throw runtime_error("It is not possible to determine the steepness of the line in a pres: " + pp(is_shallow) +". Gradient is " + pp(eq.c_j) + ". ");
      }
    }
    else if (eq.eqninf_term_present)
    {
      flat_lines.push_back(eq);
    }

    if (!eq.c_j_term_present)
    {
      if (m_defined)
      {
        if (minimal_fixed_point)
        {
           m = or_(m, eq.f_j);
        }
        else
        {
           m = and_(m, eq.f_j);
        }
      }
      else
      {
        m_defined=true;
        m = eq.f_j;
      }
    }
  }
}

void conjunction_disjunction_f_j(pres_expression& result, 
                                 bool& result_defined,
                                 std::vector< linear_fixed_point_equation >& l,
                                 const bool is_conjunction)
{
  for(const linear_fixed_point_equation& eq: l)
  {
    if (eq.f_j_term_present)
    {
      if (result_defined)
      {
        if (is_conjunction)
        {
          result = and_(result, eq.f_j);
        }
        else
        {
          result = or_(result, eq.f_j);
        }
      }
      else 
      {
        result = eq.f_j;
        result_defined=true;
      }
    }
  }
}

pres_expression conjunction_disjunction_f_j(std::vector< linear_fixed_point_equation >& l1,
                                            std::vector< linear_fixed_point_equation >& l2,
                                            std::vector< linear_fixed_point_equation >& l3,
                                            const bool is_conjunction)
{
  pres_expression result=(is_conjunction?true_():false_());
  bool result_defined=false;
  conjunction_disjunction_f_j(result, result_defined, l1, is_conjunction);
  conjunction_disjunction_f_j(result, result_defined, l2, is_conjunction);
  conjunction_disjunction_f_j(result, result_defined, l3, is_conjunction);
  return result;
}

pres_expression disjunction_infinity_cj_prime(std::vector< linear_fixed_point_equation >& l1,
                                              std::vector< linear_fixed_point_equation >& l2, 
                                              std::vector< linear_fixed_point_equation >& l3)
{
  for(const linear_fixed_point_equation& eq: l1)
  {
    if (eq.eqninf_term_present)
    {
      return true_();
    }
  }
  for(const linear_fixed_point_equation& eq: l2)
  {
    if (eq.eqninf_term_present)
    {
      return true_();
    }
  }
  for(const linear_fixed_point_equation& eq: l3)
  {
    if (eq.eqninf_term_present)
    {
      return true_();
    }
  }

  return false_();
}

pres_expression disjunction_cj_fj(std::vector< linear_fixed_point_equation >& l)
{
  pres_expression result=false_();
  bool result_defined=false;
  for(const linear_fixed_point_equation& eq: l)
  {
    data::data_expression constant = data::sort_real::divides(data::sort_real::real_one(),
                                                              data::sort_real::minus(data::sort_real::real_one(), eq.c_j));
    if (result_defined)
    {
      result = or_(result, const_multiply(constant, eq.f_j));
    }
    else 
    {
      result = const_multiply(constant, eq.f_j);
      result_defined=true;
    }
  }
  return result;
}

pres_expression conjunction_cj_fj(std::vector< linear_fixed_point_equation >& l)
{
  pres_expression result=true_();
  bool result_defined=false;
  for(const linear_fixed_point_equation& eq: l)
  {
    if (!eq.eqninf_term_present)
    {
      data::data_expression constant = data::sort_real::divides(data::sort_real::real_one(),
                                                                data::sort_real::minus(data::sort_real::real_one(), eq.c_j));
      if (result_defined)
      {
        result = or_(result, const_multiply(constant, eq.f_j));
      }
      else
      {
        result = const_multiply(constant, eq.f_j);
        result_defined=true;
      }
    }
  }
  return result;
}


pres_expression disjunction_fj_cj(std::vector< linear_fixed_point_equation >& l, const pres_expression& U, const data::rewriter& rewriter)
{
  pres_expression result=false_();
  bool result_defined=false;
  for(const linear_fixed_point_equation& eq: l)
  {
    pres_expression disjunct = false_();
    assert(eq.c_j_term_present);
    data::data_expression is_c_j_equal_one = rewriter(data::equal_to(eq.c_j, data::sort_real::real_one()));
    if (is_true(is_c_j_equal_one))
    {
      disjunct = data::sort_real::real_zero();        
    }
    else if (is_false(is_c_j_equal_one))
    {
      optimized_const_multiply(disjunct, rewriter(data::sort_real::minus(eq.c_j,data::sort_real::real_one())), U);
    }
    else
    {
      throw runtime_error("Fail to determine whether inclination equals 1 of " + pp(eq.c_j) + "(reason: " + pp(is_c_j_equal_one) + ").");
    }

    if (eq.f_j_term_present)
    {
      optimized_plus(disjunct, eq.f_j, disjunct);
    } 

    if (result_defined)
    {
      result = or_(result, disjunct);
    }
    else
    {
      result = disjunct;
      result_defined=true;
    }
  }
  return result;
}

pres_expression conjunction_fj_cj(std::vector< linear_fixed_point_equation >& l, const pres_expression& U, const data::rewriter& rewriter)
{
  pres_expression result=true_();
  bool result_defined=false;
  for(const linear_fixed_point_equation& eq: l)
  {
    if (!eq.eqninf_term_present)
    {
      assert(eq.c_j_term_present);
      pres_expression conjunct;
      data::data_expression is_c_j_equal_one = rewriter(data::equal_to(eq.c_j, data::sort_real::real_one()));
      if (is_true(is_c_j_equal_one))
      {   
        conjunct = data::sort_real::real_zero();
      }
      else if (is_false(is_c_j_equal_one))
      { 
        optimized_const_multiply(conjunct, rewriter(data::sort_real::minus(eq.c_j,data::sort_real::real_one())), U);
      }
      else
      {
        throw runtime_error("Fail to determine whether inclination equals 1 of " + pp(eq.c_j) + "(reason: " + pp(is_c_j_equal_one) + ").");
      }


      if (eq.f_j_term_present)
      {
        conjunct = plus(eq.f_j, conjunct);
      }
      if (result_defined)
      {
        result = and_(result, conjunct);
      }
      else
      {
        result = conjunct;
        result_defined=true;
      }
    }
  }
  return result;
}

pres_expression solve_fixed_point_inner(const propositional_variable& v, 
                                        const pres_expression& t, 
                                        const data::rewriter& rewriter, 
                                        const bool minimal_fixed_point)
{
  std::vector< linear_fixed_point_equation > lines; // equations c_j X + c'_j*eqninf(X) + f_j  with 0<c_j<1
  /* Here is is assumed that t is a disjunction of terms */
  collect_lines(lines, v, t, minimal_fixed_point);
 
  pres_expression m = false_();;
  std::vector< linear_fixed_point_equation > shallow_lines;     // lines with gradient between 0 and 1. 
  std::vector< linear_fixed_point_equation > steep_lines;       // lines with gradient >=1.
  std::vector< linear_fixed_point_equation > flat_lines;        // lines with gradient 0. eqninf(v) term is present. 
  collect_m_and_split_lines(lines, m, shallow_lines, steep_lines, flat_lines, rewriter, minimal_fixed_point);

  /* XXXXXX FINISH SOLVING ****/

  if (minimal_fixed_point)
  {
    pres_expression U;
    optimized_or(U, m, disjunction_cj_fj(shallow_lines));
    pres_expression cond1 = disjunction_fj_cj(steep_lines, U, rewriter);
    pres_expression cond2 = disjunction_infinity_cj_prime(shallow_lines, steep_lines, flat_lines);

    pres_expression eqinf_cond;
    optimized_eqinf(eqinf_cond, conjunction_disjunction_f_j(shallow_lines, steep_lines, flat_lines, false));
    pres_expression eqninf_m;
    optimized_eqninf(eqninf_m, m);
// std::cerr << "OPTIMIZED EQNINF " << eqninf_m << "      " << m << "\n";
    pres_expression cond4;
    optimized_or(cond4, cond1, cond2);
    pres_expression exp1;
    optimized_condeq(exp1, cond4, U, true_());
    pres_expression exp2;
    optimized_condeq(exp2, eqninf_m, false_(), exp1);
    pres_expression solution;
    optimized_condeq(solution, eqinf_cond, exp2, true_());
    pres_expression rewritten_solution=simplify_data_rewriter(rewriter)(solution);
    return rewritten_solution;
  }
  else // Maximal fixed point
  {
    pres_expression U; 
    optimized_and(U, m, conjunction_cj_fj(shallow_lines));

    pres_expression cond1 = conjunction_fj_cj(steep_lines, U, rewriter);

    pres_expression eqinf_m;
    optimized_eqinf(eqinf_m, m);
    pres_expression cond1_;
    optimized_condsm(cond1_, cond1, false_(), U);
    pres_expression solution;
    optimized_condeq(solution, eqinf_m, cond1_, true_());
    pres_expression rewritten_solution=simplify_data_rewriter(rewriter)(solution);
    return rewritten_solution;
  }
}

const pres_expression solve_single_equation(const fixpoint_symbol& f, const propositional_variable& v, const pres_expression& t, const data::rewriter& rewriter)
{
  pres_expression aux;
  if (is_condsm(t) && f==pbes_system::fixpoint_symbol::mu())
  { 
    const condsm tc = atermpp::down_cast<condsm>(t);
    pres_expression solution_arg2 = solve_single_equation(f, v, tc.arg2(), rewriter);
    
    const bool conjunctive_normal_form = true;
    push_or_inside(aux, tc.arg2(), tc.arg3(), conjunctive_normal_form);

    pres_expression solution_arg2_or_arg3 = solve_single_equation(f, v, aux, rewriter);
    
    pres_expression new_condition;
    substitute_pres_equation_builder variable_substituter(v, solution_arg2);
    variable_substituter.apply(new_condition, tc.arg1());
    return condsm(new_condition, solution_arg2, solution_arg2_or_arg3);
  }
  else if (is_condsm(t) && f==pbes_system::fixpoint_symbol::nu())
  { 
    const condsm tc = atermpp::down_cast<condsm>(t);
    pres_expression solution_arg2 = solve_single_equation(f, v, tc.arg2(), rewriter);
    pres_expression solution_arg3 = solve_single_equation(f, v, tc.arg3(), rewriter);
    
    pres_expression new_condition;
    substitute_pres_equation_builder variable_substituter(v, or_(solution_arg2, solution_arg3));
    variable_substituter.apply(new_condition, tc.arg1());
    
    return condsm(new_condition, solution_arg2, solution_arg3);
  }
  else if (is_condeq(t) && f==pbes_system::fixpoint_symbol::mu())
  { 
    const condeq tc = atermpp::down_cast<condeq>(t);
    pres_expression solution_arg2 = solve_single_equation(f, v, tc.arg2(), rewriter);
    pres_expression solution_arg3 = solve_single_equation(f, v, tc.arg3(), rewriter);
    
    pres_expression new_condition;
    substitute_pres_equation_builder variable_substituter(v, and_(solution_arg2, solution_arg3));
    variable_substituter.apply(new_condition, tc.arg1());
    
    return condeq(new_condition, solution_arg2, solution_arg3);
  }
  if (is_condeq(t) && f==pbes_system::fixpoint_symbol::nu())
  { 
    const condeq tc = atermpp::down_cast<condeq>(t);
    pres_expression solution_arg2 = solve_single_equation(f, v, tc.arg2(), rewriter);
    
    const bool conjunctive_normal_form = false;
    push_and_inside(aux, tc.arg2(), tc.arg3(), conjunctive_normal_form);

    pres_expression solution_arg2_or_arg3 = solve_single_equation(f, v, aux, rewriter);
    
    pres_expression new_condition;
    substitute_pres_equation_builder variable_substituter(v, solution_arg2);
    variable_substituter.apply(new_condition, tc.arg1());
    
    return condeq(new_condition, solution_arg2, solution_arg2_or_arg3);
  }
  else if (is_and(t) && f==pbes_system::fixpoint_symbol::mu())
  {
    const and_ tc = atermpp::down_cast<and_>(t);
    pres_expression solution_left = solve_single_equation(f, v, tc.left(), rewriter);
    pres_expression solution_right = solve_single_equation(f, v, tc.right(), rewriter);
    
    return and_(solution_left, solution_right);
  }
  else if (is_or(t) && f==pbes_system::fixpoint_symbol::nu())
  {
    const or_ tc = atermpp::down_cast<or_>(t);
    pres_expression solution_left = solve_single_equation(f, v, tc.left(), rewriter);
    pres_expression solution_right = solve_single_equation(f, v, tc.right(), rewriter);
    
    return or_(solution_left, solution_right);
  }
  else if (f==pbes_system::fixpoint_symbol::mu())
  {
    // Solve minimal fixed point. 
    return solve_fixed_point_inner(v, t, rewriter, true);
  }
  else 
  {
    assert(f==pbes_system::fixpoint_symbol::nu());
    // Solve maximal fixed point. 
    return solve_fixed_point_inner(v, t, rewriter, false);
  }
}

} // namespace detail



class res_conjunctive_disjunctive_normal_form_builder: public pres_expression_builder <res_conjunctive_disjunctive_normal_form_builder>
{
protected:
  bool m_conjunctive_normal_form;
  bool m_negate=false;

public:
  typedef pres_expression_builder<res_conjunctive_disjunctive_normal_form_builder> super;
  using super::apply;

  res_conjunctive_disjunctive_normal_form_builder(const bool conjunctive)
   : m_conjunctive_normal_form(conjunctive)
  {}

  template <class T>
  void apply(T& result, const pres_system::propositional_variable_instantiation& x)
  {
    pres_system::make_propositional_variable_instantiation(result, x.name(), [&](data::data_expression_list& result){ apply(result, x.parameters()); });
    if (m_negate)
    {
      pres_system::optimized_minus(result, result);
    }
  }

  template <class T>
  void apply(T& result, const pres_system::and_& x)
  {
    pres_expression aux1;
    pres_expression aux2;
    apply(aux1, x.left());
    apply(aux2, x.right());
    detail::push_and_inside(result, aux1, aux2, m_conjunctive_normal_form);
  }

  template <class T>
  void apply(T& result, const pres_system::or_& x)
  {
    pres_expression aux1;
    pres_expression aux2;
    apply(aux1, x.left());
    apply(aux2, x.right());
    detail::push_or_inside(result, aux1, aux2, m_conjunctive_normal_form);
  }

  template <class T>
  void apply(T& result, const pres_system::imp& x)
  {
    pres_expression aux;
    optimized_minus(aux, x.left());
    optimized_or(aux, aux, x.right());
    apply(result,aux);
  }

  template <class T>
  void apply(T& result, const pres_system::plus& x)
  {
// std::cerr << "NF PLUS " << static_cast<pres_expression>(x) << "\n";
    pres_expression aux1, aux2;
    apply(aux1, x.left());
    apply(aux2, x.right());
    detail::push_plus_inside(result, aux1, aux2, m_conjunctive_normal_form);
// std::cerr << "NF PLUS " << static_cast<pres_expression>(x) << " ---> " << result << "\n";
  }

  template <class T>
  void apply(T& result, const pres_system::const_multiply& x)
  {
    pres_expression aux;
    apply(aux, x.right());
    const data::data_expression& constant=x.left();
    detail::push_constant_inside(result, constant, aux);
  }

  template <class T>
  void apply(T& result, const pres_system::const_multiply_alt& x)
  {
    pres_expression aux;
    apply(aux, x.left());
    const data::data_expression& constant=x.right();
    detail::push_constant_inside(result, constant, aux);
  }

  template <class T>
  void apply(T&, const pres_system::minall& x)
  {
    throw runtime_error("Cannot normalise and solve a res with a minall operator: " + pp(static_cast<pres_expression>(x)) + ".");
  }

  template <class T>
  void apply(T&, const pres_system::maxall& x)
  {
    throw runtime_error("Cannot normalise and solve a res with a maxall operator: " + pp(static_cast<pres_expression>(x)) + ".");
  }

  template <class T>
  void apply(T&, const pres_system::sum& x)
  {
    throw runtime_error("Cannot normalise and solve a res with a sum operator: " + pp(static_cast<pres_expression>(x)) + ".");
  }

  template <class T>
  void apply(T& result, const pres_system::eqinf& x)
  {
    pres_system::make_eqinf(result, [&](pres_expression& result){ apply(result, x.operand()); });
  }

  template <class T>
  void apply(T& result, const pres_system::eqninf& x)
  {
    pres_system::make_eqninf(result, [&](pres_expression& result){ apply(result, x.operand()); });
  }
}; 

/// \brief An algorithm that takes a res, i.e. a pres with propositional variables without parameters
///        and solves it by Gauss elimination.

class ressolve_by_gauss_elimination_algorithm
{
  protected:
    pressolve_options m_options;
    data::rewriter m_datar;    // data_rewriter
    pres m_input_pres;
    enumerate_quantifiers_rewriter m_R;   // The rewriter.

    data::rewriter construct_rewriter(const pres& presspec)
    {
      if (m_options.remove_unused_rewrite_rules)
      {
        std::set<data::function_symbol> used_functions = pres_system::find_function_symbols(presspec);
        used_functions.insert(data::less(data::sort_real::real_()));
        used_functions.insert(data::sort_real::divides(data::sort_real::real_(),data::sort_real::real_()));
        used_functions.insert(data::sort_real::times(data::sort_real::real_(),data::sort_real::real_()));
        used_functions.insert(data::sort_real::plus(data::sort_real::real_(),data::sort_real::real_()));
        used_functions.insert(data::sort_real::minus(data::sort_real::real_(),data::sort_real::real_()));
        used_functions.insert(data::sort_real::minimum(data::sort_real::real_(),data::sort_real::real_()));
        used_functions.insert(data::sort_real::maximum(data::sort_real::real_(),data::sort_real::real_()));
        return data::rewriter(presspec.data(),
                              data::used_data_equation_selector(presspec.data(), used_functions, presspec.global_variables()),
                              m_options.rewrite_strategy);
      }
      else
      {
        return data::rewriter(presspec.data(), m_options.rewrite_strategy);
      }
    }

  public:
    ressolve_by_gauss_elimination_algorithm(
      const pressolve_options& options,
      const pres& input_pres
    ) 
     : m_options(options),
       m_datar(construct_rewriter(input_pres)),
       m_input_pres(input_pres),
       m_R(m_datar,input_pres.data())
    {}

    const pres_expression run()
    {
      std::vector<pres_equation> res_equations(m_input_pres.equations().begin(), m_input_pres.equations().end());
      assert(res_equations.size()>0);
      // Take care that the first equation has the initial variable at the lhs.
      if (res_equations.front().variable().name()!=m_input_pres.initial_state().name())
      {
        core::identifier_string new_name("initial_variable$$"); // Name should not be seen externally.
        res_equations.insert(res_equations.begin(),
                             pres_equation(res_equations.front().symbol(),
                                           propositional_variable(new_name, data::variable_list()),
                                           m_input_pres.initial_state()));
         m_input_pres.initial_state() = propositional_variable_instantiation(new_name, data::data_expression_list());
      }

      atermpp::indexed_set<core::identifier_string> variable_names;
      pres_expression result;
      res_conjunctive_disjunctive_normal_form_builder conjunctive_normal_form_builder(true);
      res_conjunctive_disjunctive_normal_form_builder disjunctive_normal_form_builder(false);
      for(pres_equation& e: res_equations)
      {
        variable_names.insert(e.variable().name());
      }

      for(std::vector<pres_equation>::reverse_iterator equation_it=res_equations.rbegin(); equation_it!=res_equations.rend(); equation_it++)
      {
        if (equation_it->symbol().is_mu())
        {
// std::cerr << "IN1 " << equation_it->formula() << "\n";
// std::cerr << "IN2 " << m_R(equation_it->formula()) << "\n";
          conjunctive_normal_form_builder.apply(result, m_R(equation_it->formula()));
// std::cerr << "OUT " << result << "\n";
        }
        else
        {
          disjunctive_normal_form_builder.apply(result, m_R(equation_it->formula()));
        }

        pres_expression solution = detail::solve_single_equation(equation_it->symbol(),
                                                                 equation_it->variable(),
                                                                 result,
                                                                 m_datar);
        mCRL2log(log::debug) << "Solving    " << equation_it->symbol() << " " << equation_it->variable() << " = " << equation_it->formula() << "\n";
        mCRL2log(log::debug) << "Norm. Form " << equation_it->symbol() << " " << equation_it->variable() << " = " << result << "\n";  
        equation_it->formula() = solution;
        mCRL2log(log::debug) << "Solution   " << equation_it->symbol() << " " << equation_it->variable() << " = " << equation_it->formula() << "\n";

        substitute_pres_equation_builder substitute_pres_equation(equation_it->variable(), solution);
       
        for(std::vector<pres_equation>::iterator substitution_equation_it=res_equations.begin();
                                                 substitution_equation_it!=equation_it.base(); 
                                                 substitution_equation_it++)
        {
          substitute_pres_equation.apply(result, substitution_equation_it->formula());
          substitution_equation_it->formula() = result;
        }
      }
      return m_R(res_equations.front().formula());
    } 
};

} // namespace pres_system

} // namespace mcrl2

#endif // MCRL2_PRES_RESSOLVE_H
