// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/rewriters/custom_enumerate_quantifiers_rewriter.h
/// \brief add your file description here.

#ifndef MCRL2_PBES_REWRITERS_CUSTOM_ENUMERATE_QUANTIFIERS_REWRITER_H
#define MCRL2_PBES_REWRITERS_CUSTOM_ENUMERATE_QUANTIFIERS_REWRITER_H

#include <numeric>
#include <set>
#include <utility>
#include <deque>
#include <sstream>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include "mcrl2/core/detail/print_utility.h"
#include "mcrl2/data/enumerator.h"
#include "mcrl2/data/data_specification.h"
#include "mcrl2/pbes/rewriters/simplify_rewriter.h"
#include "mcrl2/pbes/replace.h"
#include "mcrl2/utilities/optimized_boolean_operators.h"
#include "mcrl2/utilities/sequence.h"
#include "mcrl2/utilities/detail/join.h"

namespace mcrl2 {

namespace pbes_system {

namespace detail {

// inserts elements of c into s
template <typename T, typename Container>
void set_insert(std::set<T>& s, const Container& c)
{
  for (auto i = c.begin(); i != c.end(); ++i)
  {
    s.insert(*i);
  }
}

// removes elements of c from s
template <typename T, typename Container>
void set_remove(std::set<T>& s, const Container& c)
{
  for (auto i = c.begin(); i != c.end(); ++i)
  {
    s.erase(*i);
  }
}

/// \brief Computes the subset with variables of finite sort and infinite.
// TODO: this should be done more efficiently, by avoiding aterm lists
/// \param variables A sequence of data variables
/// \param data A data specification
/// \param finite_variables A sequence of data variables
/// \param infinite_variables A sequence of data variables
inline
void split_finite_variables(data::variable_list variables, const data::data_specification& data, data::variable_list& finite_variables, data::variable_list& infinite_variables)
{
  std::vector<data::variable> finite;
  std::vector<data::variable> infinite;
  for (auto i = variables.begin(); i != variables.end(); ++i)
  {
    if (data.is_certainly_finite(i->sort()))
    {
      finite.push_back(*i);
    }
    else
    {
      infinite.push_back(*i);
    }
  }
  finite_variables = data::variable_list(finite.begin(), finite.end());
  infinite_variables = data::variable_list(infinite.begin(), infinite.end());
}

template <typename PbesRewriter>
class quantifier_enumerator
{
  protected:
    PbesRewriter& pbesr;
    const data::data_enumerator& datae;

    typedef typename core::term_traits<pbes_expression> tr;

    /// Exception that is used as an early escape of the foreach_sequence algorithm.
    struct stop_early
      {};

    /// Joins a sequence of pbes expressions with operator and
    struct join_and
    {
      /// \brief Returns the conjunction of a sequence of pbes expressions
      /// \param first Start of a sequence of pbes expressions
      /// \param last End of a sequence of pbes expressions
      /// \return The conjunction of the expressions
      template <typename FwdIt>
      pbes_expression operator()(FwdIt first, FwdIt last) const
      {
        return std::accumulate(first, last, core::term_traits<pbes_expression>::true_(), &utilities::optimized_and<pbes_expression>);
      }
    };

    /// Joins a sequence of pbes expressions with operator or
    struct join_or
    {
      /// \brief Returns the disjunction of a sequence of pbes expressions
      /// \param first Start of a sequence of pbes expressions
      /// \param last End of a sequence of pbes expressions
      /// \return The disjunction of the expressions
      template <typename FwdIt>
      pbes_expression operator()(FwdIt first, FwdIt last) const
      {
        return std::accumulate(first, last, core::term_traits<pbes_expression>::false_(), &utilities::optimized_or<pbes_expression>);
      }
    };

    /// The assign operation used to create sequences in the foreach_sequence algorithm
    template <typename SubstitutionFunction>
    struct sequence_assign
    {
      typedef typename SubstitutionFunction::variable_type variable_type;
      typedef typename SubstitutionFunction::expression_type pbes_expression;

      SubstitutionFunction& sigma_;

      sequence_assign(SubstitutionFunction& sigma)
        : sigma_(sigma)
      {}

      /// \brief Function call operator
      /// \param v A variable
      /// \param t A term
      void operator()(const data::variable& v, const pbes_expression& t)
      {
        sigma_[v] = t;
      }
    };

    /// The action that is triggered for each sequence generated by the
    /// foreach_sequence algorithm. It is invoked for every sequence of
    /// substitutions of the set Z in the algorithm.
    template <typename SubstitutionFunction, typename StopCriterion>
    struct sequence_action
    {
      std::set<pbes_expression>& A_;
      PbesRewriter& r_;
      const pbes_expression& phi_;
      SubstitutionFunction& sigma_;
      const std::set<data::variable>& v_;
      bool& is_constant_;
      StopCriterion stop_;

      /// \brief Determines if the unordered sequences s1 and s2 have an empty intersection
      /// \param s1 A sequence
      /// \param s2 A sequence
      /// \return True if the intersection of s1 and s2 is empty
      template <typename Sequence, typename Set>
      bool empty_intersection(const Sequence& s1, const Set& s2)
      {
        for (auto i = s1.begin(); i != s1.end(); ++i)
        {
          if (s2.find(*i) != s2.end())
          {
            return false;
          }
        }
        return true;
      }

      sequence_action(std::set<pbes_expression>& A,
                      PbesRewriter& r,
                      const pbes_expression& phi,
                      SubstitutionFunction& sigma,
                      const std::set<data::variable>& v,
                      bool& is_constant,
                      StopCriterion stop
                     )
        : A_(A), r_(r), phi_(phi), sigma_(sigma), v_(v), is_constant_(is_constant), stop_(stop)
      {}

      /// \brief Function call operator
      void operator()()
      {
        pbes_expression c = r_(phi_, sigma_);
        std::set<data::variable> FV_c = pbes_system::find_free_variables(c);

        mCRL2log(log::verbose) << "        Z = Z + " << c << " sigma = " << data::print_substitution(sigma_) << " dependencies = " << core::detail::print_list(v_) << std::endl;
        if (stop_(c))
        {
          throw stop_early();
        }
        else if (empty_intersection(FV_c, v_))
        {
          mCRL2log(log::verbose) << "        A = A + " << pbes_system::pp(c) << std::endl;
          A_.insert(c);
        }
        else
        {
          is_constant_ = false;
        }
      }
    };

    /// Convenience function for generating a sequence action
    template <typename SubstitutionFunction, typename StopCriterion>
    sequence_action<SubstitutionFunction, StopCriterion>
    make_sequence_action(std::set<pbes_expression>& A,
                         PbesRewriter& r,
                         const pbes_expression& phi,
                         SubstitutionFunction& sigma,
                         const std::set<data::variable>& v,
                         bool& is_constant,
                         StopCriterion stop
                        )
    {
      return sequence_action<SubstitutionFunction, StopCriterion>(A, r, phi, sigma, v, is_constant, stop);
    }

    /// \brief Prints debug information to standard error
    /// \param x A sequence of variables
    /// \param phi A term
    /// \param sigma A substitution function
    /// \param stop_value A term
    template <typename SubstitutionFunction>
    void print_arguments(data::variable_list x, const pbes_expression& phi, SubstitutionFunction& sigma, pbes_expression stop_value) const
    {
      mCRL2log(log::verbose) << "<enumerate>"
                             << (tr::is_false(stop_value) ? "forall " : "exists ")
                             << x << ". "
                             << phi
                             << data::print_substitution(sigma) << std::endl;
    }

    /// \brief Returns a string representation of D[i]
    /// \param Di A sequence of data terms
    /// \param i A positive integer
    /// \return A string representation of D[i]
    std::string print_D_element(const std::vector<data::data_expression_with_variables>& Di, std::size_t i) const
    {
      std::ostringstream out;
      out << "D[" << i << "] = " << core::detail::print_list(Di) << std::endl;
      return out.str();
    }

    /// \brief Prints debug information to standard error
    /// \param D The sequence D of the algorithm
    void print_D(const std::vector<std::vector<data::data_expression_with_variables> >& D) const
    {
      for (size_t i = 0; i < D.size(); i++)
      {
        mCRL2log(log::verbose) << "  " << print_D_element(D[i], i);
      }
    }

    /// \brief Returns a string representation of a todo list element
    /// \param e A todo list element
    /// \return A string representation of a todo list element
    std::string print_todo_list_element(const boost::tuple<data::variable, data::data_expression_with_variables, std::size_t>& e) const
    {
      // const data::variable& xk = boost::get<0>(e);
      const data::data_expression_with_variables& y = boost::get<1>(e);
      std::size_t k = boost::get<2>(e);
      return "(" + data::pp(y) + ", " + boost::lexical_cast<std::string>(k) + ")";
    }

    /// \brief Prints a todo list to standard error
    /// \param todo A todo list
    void print_todo_list(const std::deque<boost::tuple<data::variable, data::data_expression_with_variables, std::size_t> >& todo) const
    {
      mCRL2log(log::verbose) << "  todo = [";
      for (auto i = todo.begin(); i != todo.end(); ++i)
      {
        mCRL2log(log::verbose) << (i == todo.begin() ? "" : ", ") << print_todo_list_element(*i);
      }
      mCRL2log(log::verbose) << "]" << std::endl;
    }

    template <typename SubstitutionFunction, typename VariableMap>
    void redo_substitutions(SubstitutionFunction& sigma, const VariableMap& v)
    {
      for (auto i = v.begin(); i != v.end(); ++i)
      {
        sigma[i->first] = i->second;
      }
    }

    template <typename SubstitutionFunction, typename StopCriterion, typename PbesTermJoinFunction>
    pbes_expression enumerate(data::variable_list x,
                        const pbes_expression& phi,
                        SubstitutionFunction& sigma,
                        StopCriterion stop,
                        pbes_expression stop_value,
                        PbesTermJoinFunction join
                       )
    {
      // Undo substitutions to quantifier variables
      std::map<data::variable, data::data_expression_with_variables> undo;
      for (auto i = x.begin(); i != x.end(); ++i)
      {
        pbes_expression sigma_i = sigma(*i);
        if (sigma_i != *i)
        {
          undo[*i] = sigma_i;
          sigma[*i] = *i;
        }
      }

      print_arguments(x, phi, sigma, stop_value);
      pbes_expression Rphi = pbesr(phi, sigma);
      if (tr::is_constant(Rphi))
      {
        redo_substitutions(sigma, undo);
        return Rphi;
      }

      std::set<pbes_expression> A;
      std::vector<std::vector<data::data_expression_with_variables> > D;
      std::set<data::variable> dependencies;

      // For an element (v, t, k) of todo, we have the invariant v == x[k].
      // The variable v is stored for efficiency reasons, it avoids the lookup x[k].
      std::deque<boost::tuple<data::variable, data::data_expression_with_variables, std::size_t> > todo;

      // initialize D and todo
      std::size_t j = 0;
      for (auto i = x.begin(); i != x.end(); ++i)
      {
        data::data_expression_with_variables t = core::term_traits<data::data_expression_with_variables>::variable2term(*i);
        D.push_back(std::vector<data::data_expression_with_variables>(1, t));
        todo.push_back(boost::make_tuple(*i, t, j++));
        set_insert(dependencies, t.variables());
      }

      try
      {
        while (!todo.empty())
        {
          boost::tuple<data::variable, data::data_expression_with_variables, std::size_t> front = todo.front();
          print_D(D);
          print_todo_list(todo);
          mCRL2log(log::verbose) << "    (y, k) = " << print_todo_list_element(front) << std::endl;
          todo.pop_front();
          const data::variable& xk = boost::get<0>(front);
          const data::data_expression_with_variables& y = boost::get<1>(front);
          std::size_t k = boost::get<2>(front);
          bool is_constant = true;

          D[k].erase(std::find(D[k].begin(), D[k].end(), y));
          set_remove(dependencies, y.variables());

          // save D[k] in variable Dk, as a preparation for the foreach_sequence algorithm
          std::vector<data::data_expression_with_variables> Dk = D[k];
          std::vector<data::data_expression_with_variables> z = datae.enumerate(y);
          for (auto i = z.begin(); i != z.end(); ++i)
          {
            mCRL2log(log::verbose) << "      e = " << data::pp(*i) << std::endl;
            set_insert(dependencies, i->variables());
            sigma[xk] = *i;
            D[k].clear();
            D[k].push_back(*i);
            utilities::foreach_sequence(D,
                                   x.begin(),
                                   make_sequence_action(A, pbesr, phi, sigma, dependencies, is_constant, stop),
                                   sequence_assign<SubstitutionFunction>(sigma)
                                  );
            if (!is_constant)
            {
              Dk.push_back(*i);
              mCRL2log(log::verbose) << "        " << print_D_element(Dk, k) << std::endl;
              if (!core::term_traits<data::data_expression_with_variables>::is_constant(*i))
              {
                todo.push_back(boost::make_tuple(xk, *i, k));
              }
              else
              {
                set_remove(dependencies, i->variables());
              }
            }
          }

          // restore D[k]
          D[k] = Dk;
        }
      }
      catch (stop_early&)
      {
        // remove the added substitutions from sigma
        for (auto j = x.begin(); j != x.end(); ++j)
        {
          sigma[*j] = *j; // erase *j
        }
        mCRL2log(log::verbose) << "<return>stop early: " << pbes_system::pp(stop_value) << std::endl;
        redo_substitutions(sigma, undo);
        return stop_value;
      }

      // remove the added substitutions from sigma
      for (auto i = x.begin(); i != x.end(); ++i)
      {
        sigma[*i] = *i; // erase *i
      }
      pbes_expression result = join(A.begin(), A.end());
      mCRL2log(log::verbose) << "<return> " << pbes_system::pp(result) << std::endl;
      redo_substitutions(sigma, undo);
      return result;
    }

  public:
    quantifier_enumerator(PbesRewriter& r, const data::data_enumerator& e)
      : pbesr(r), datae(e)
    {}

    /// \brief Enumerates a universal quantification
    /// \param x A sequence of variables
    /// \param phi A term
    /// \param sigma A substitution function
    /// \return The enumeration result
    template <typename SubstitutionFunction>
    pbes_expression enumerate_universal_quantification(const data::variable_list& x, const pbes_expression& phi, SubstitutionFunction& sigma)
    {
      return enumerate(x, phi, sigma, tr::is_false, tr::false_(), join_and());
    }

    /// \brief Enumerates an existential quantification
    /// \param x A sequence of variables
    /// \param phi A term
    /// \param sigma A substitution function
    /// \return The enumeration result
    template <typename SubstitutionFunction>
    pbes_expression enumerate_existential_quantification(data::variable_list x, pbes_expression phi, SubstitutionFunction& sigma)
    {
      return enumerate(x, phi, sigma, tr::is_true, tr::true_(), join_or());
    }
};

// Simplifying PBES rewriter that eliminates quantifiers using enumeration.
/// \param SubstitutionFunction This must be a MapSubstitution.
template <typename Derived, typename DataRewriter, typename SubstitutionFunction>
struct custom_enumerate_quantifiers_builder: public simplify_data_rewriter_builder<Derived, DataRewriter, SubstitutionFunction>
{
  typedef simplify_data_rewriter_builder<Derived, DataRewriter, SubstitutionFunction> super;
  using super::enter;
  using super::leave;
  using super::operator();
  using super::sigma;

  typedef custom_enumerate_quantifiers_builder<Derived, DataRewriter, SubstitutionFunction> self;
  typedef core::term_traits<pbes_expression> tr;

  const data::data_enumerator& m_data_enumerator;

  /// If true, quantifier variables of infinite sort are enumerated.
  bool m_enumerate_infinite_sorts;

  /// \brief Constructor.
  /// \param r A data rewriter
  /// \param enumerator A data enumerator
  /// \param enumerate_infinite_sorts If true, quantifier variables of infinite sort are enumerated as well
  custom_enumerate_quantifiers_builder(const data::rewriter& R, SubstitutionFunction& sigma, const data::data_enumerator& enumerator, bool enumerate_infinite_sorts = true)
    : super(R, sigma), m_data_enumerator(enumerator), m_enumerate_infinite_sorts(enumerate_infinite_sorts)
  { }

  Derived& derived()
  {
    return static_cast<Derived&>(*this);
  }

  pbes_expression operator()(const forall& x)
  {
    pbes_expression result;
    if (m_enumerate_infinite_sorts)
    {
      result = quantifier_enumerator<self>(*this, m_data_enumerator).enumerate_universal_quantification(x.variables(), x.body(), sigma);
    }
    else
    {
      data::variable_list finite;
      data::variable_list infinite;
      split_finite_variables(x.variables(), m_data_enumerator.data(), finite, infinite);
      if (finite.empty())
      {
        result = utilities::optimized_forall(infinite, derived()(x.body()));
      }
      else
      {
        result = utilities::optimized_forall_no_empty_domain(infinite, quantifier_enumerator<self>(*this, m_data_enumerator).enumerate_universal_quantification(finite, x.body(), sigma));
      }
    }
    return result;
  }

  pbes_expression operator()(const exists& x)
  {
    pbes_expression result;
    if (m_enumerate_infinite_sorts)
    {
      result = quantifier_enumerator<self>(*this, m_data_enumerator).enumerate_existential_quantification(x.variables(), x.body(), sigma);
    }
    else
    {
      data::variable_list finite;
      data::variable_list infinite;
      split_finite_variables(x.variables(), m_data_enumerator.data(), finite, infinite);
      if (finite.empty())
      {
        result = utilities::optimized_exists(infinite, derived()(x.body()));
      }
      else
      {
        result = utilities::optimized_exists_no_empty_domain(infinite, quantifier_enumerator<self>(*this, m_data_enumerator).enumerate_existential_quantification(finite, x.body(), sigma));
      }
    }
    return result;
  }

  // TODO: this function should be removed
  pbes_expression operator()(const pbes_expression& x, SubstitutionFunction&)
  {
    return (*this)(x);
  }
};

template <template <class, class, class> class Builder, class DataRewriter, class SubstitutionFunction>
struct apply_enumerate_builder: public Builder<apply_enumerate_builder<Builder, DataRewriter, SubstitutionFunction>, DataRewriter, SubstitutionFunction>
{
  typedef Builder<apply_enumerate_builder<Builder, DataRewriter, SubstitutionFunction>, DataRewriter, SubstitutionFunction> super;
  using super::enter;
  using super::leave;
  using super::operator();

  apply_enumerate_builder(const DataRewriter& R, SubstitutionFunction& sigma, const data::data_enumerator& enumerator, bool enumerate_infinite_sorts)
    : super(R, sigma, enumerator, enumerate_infinite_sorts)
  {}

#ifdef BOOST_MSVC
#include "mcrl2/core/detail/builder_msvc.inc.h"
#endif
};

template <template <class, class, class> class Builder, class DataRewriter, class SubstitutionFunction>
apply_enumerate_builder<Builder, DataRewriter, SubstitutionFunction>
make_apply_enumerate_builder(const DataRewriter& R, SubstitutionFunction& sigma, const data::data_enumerator& enumerator, bool enumerate_infinite_sorts)
{
  return apply_enumerate_builder<Builder, DataRewriter, SubstitutionFunction>(R, sigma, enumerator, enumerate_infinite_sorts);
}

} // namespace detail

/// \brief An attempt for improving the efficiency.
struct custom_enumerate_quantifiers_rewriter
{
  /// \brief A data rewriter
  data::rewriter m_rewriter;

  /// \brief A data enumerator
  data::data_enumerator m_enumerator;

  /// \brief If true, quantifier variables of infinite sort are enumerated.
  bool m_enumerate_infinite_sorts;

  typedef pbes_expression term_type;
  typedef data::variable variable_type;

  custom_enumerate_quantifiers_rewriter(const data::rewriter& R, const data::data_enumerator& enumerator, bool enumerate_infinite_sorts = true)
    : m_rewriter(R), m_enumerator(enumerator), m_enumerate_infinite_sorts(enumerate_infinite_sorts)
  {}

  pbes_expression operator()(const pbes_expression& x) const
  {
    data::mutable_map_substitution<> sigma;
    return detail::make_apply_enumerate_builder<detail::custom_enumerate_quantifiers_builder>(m_rewriter, sigma, m_enumerator, m_enumerate_infinite_sorts)(x);
  }

  template <typename SubstitutionFunction>
  pbes_expression operator()(const pbes_expression& x, SubstitutionFunction& sigma) const
  {
    return detail::make_apply_enumerate_builder<detail::custom_enumerate_quantifiers_builder>(m_rewriter, sigma, m_enumerator, m_enumerate_infinite_sorts)(x);
  }
};

} // namespace pbes_system

} // namespace mcrl2

#endif // MCRL2_PBES_REWRITERS_CUSTOM_ENUMERATE_QUANTIFIERS_REWRITER_H
