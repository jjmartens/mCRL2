// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/detail/control_flow.h
/// \brief add your file description here.

#ifndef MCRL2_PBES_DETAIL_CONTROL_FLOW_H
#define MCRL2_PBES_DETAIL_CONTROL_FLOW_H

#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include "mcrl2/data/replace.h"
#include "mcrl2/data/rewriter.h"
#include "mcrl2/data/standard.h"
#include "mcrl2/data/standard_utility.h"
#include "mcrl2/data/detail/simplify_rewrite_builder.h"
#include "mcrl2/pbes/find.h"
#include "mcrl2/pbes/rewrite.h"
#include "mcrl2/pbes/rewriter.h"
#include "mcrl2/pbes/detail/is_pfnf.h"
#include "mcrl2/pbes/detail/pfnf_pbes.h"
#include "mcrl2/pbes/detail/simplify_quantifier_builder.h"
#include "mcrl2/pbes/detail/control_flow_influence.h"
#include "mcrl2/pbes/detail/control_flow_source_dest.h"
#include "mcrl2/pbes/detail/control_flow_utility.h"
#include "mcrl2/utilities/logger.h"

namespace mcrl2 {

namespace pbes_system {

namespace detail {

// Adds some simplifications to simplify_rewrite_builder.
template <typename Term, typename DataRewriter, typename SubstitutionFunction = no_substitution>
struct control_flow_simplify_quantifier_builder: public pbes_system::detail::simplify_quantifier_builder<Term, DataRewriter, SubstitutionFunction>
{
  typedef pbes_system::detail::simplify_quantifier_builder<Term, DataRewriter, SubstitutionFunction> super;
  typedef SubstitutionFunction                                                                       argument_type;
  typedef typename super::term_type                                                                  term_type;
  typedef typename core::term_traits<term_type>::data_term_type                                      data_term_type;
  typedef typename core::term_traits<term_type>::data_term_sequence_type                             data_term_sequence_type;
  typedef typename core::term_traits<term_type>::variable_sequence_type                              variable_sequence_type;
  typedef typename core::term_traits<term_type>::propositional_variable_type                         propositional_variable_type;
  typedef core::term_traits<Term> tr;

  /// \brief Constructor.
  /// \param rewr A data rewriter
  control_flow_simplify_quantifier_builder(const DataRewriter& rewr)
    : simplify_quantifier_builder<Term, DataRewriter, SubstitutionFunction>(rewr)
  { }

  bool is_data_not(const pbes_expression& x) const
  {
    return data::is_data_expression(x) && data::sort_bool::is_not_application(x);
  }

  // replace !(y || z) by !y && !z
  // replace !(y && z) by !y || !z
  // replace !(y => z) by y || !z
  // replace y => z by !y || z
  term_type post_process(const term_type& x)
  {
    term_type result = x;
    if (tr::is_not(x))
    {
      term_type t = tr::not_arg(x);
      if (tr::is_and(t)) // x = !(y && z)
      {
        term_type y = utilities::optimized_not(tr::left(t));
        term_type z = utilities::optimized_not(tr::right(t));
        result = utilities::optimized_and(y, z);
      }
      else if (tr::is_or(t)) // x = !(y || z)
      {
        term_type y = utilities::optimized_not(tr::left(t));
        term_type z = utilities::optimized_not(tr::right(t));
        result = utilities::optimized_or(y, z);
      }
      else if (tr::is_imp(t)) // x = !(y => z)
      {
        term_type y = tr::left(t);
        term_type z = utilities::optimized_not(tr::right(t));
        result = utilities::optimized_or(y, z);
      }
      else if (is_data_not(t)) // x = !val(!y)
      {
        term_type y = data::application(t).arguments().front();
        result = y;
      }
    }
    else if (tr::is_imp(x)) // x = y => z
    {
      term_type y = utilities::optimized_not(tr::left(x));
      term_type z = tr::right(x);
      result = utilities::optimized_or(y, z);
    }
    return result;
  }

  // replace the data expression y != z by !(y == z)
  term_type visit_data_expression(const term_type& x, const data_term_type& d, SubstitutionFunction& sigma)
  {
    typedef core::term_traits<data::data_expression> tt;
    term_type result = super::visit_data_expression(x, d, sigma);
    data::data_expression t = result;
    if (data::is_not_equal_to_application(t)) // result = y != z
    {
      data::data_expression y = tt::left(t);
      data::data_expression z = tt::right(t);
      result = tr::not_(data::equal_to(y, z));
    }
    return post_process(result);
  }

  term_type visit_true(const term_type& x, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_true(x, sigma));
  }

  term_type visit_false(const term_type& x, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_false(x, sigma));
  }

  term_type visit_not(const term_type& x, const term_type& n, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_not(x, n, sigma));
  }

  term_type visit_and(const term_type& x, const term_type& left, const term_type& right, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_and(x, left, right, sigma));
  }

  term_type visit_or(const term_type& x, const term_type& left, const term_type& right, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_or(x, left, right, sigma));
  }

  term_type visit_imp(const term_type& x, const term_type& left, const term_type& right, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_imp(x, left, right, sigma));
  }

  term_type visit_forall(const term_type& x, const variable_sequence_type&  variables, const term_type&  expression, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_forall(x, variables, expression, sigma));
  }

  term_type visit_exists(const term_type& x, const variable_sequence_type&  variables, const term_type&  expression, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_exists(x, variables, expression, sigma));
  }

  term_type visit_propositional_variable(const term_type& x, const propositional_variable_type&  v, SubstitutionFunction& sigma)
  {
    return post_process(super::visit_propositional_variable(x, v, sigma));
  }
};

template <typename Term, typename DataRewriter>
class control_flow_simplifying_rewriter
{
  protected:
    DataRewriter m_rewriter;

  public:
    typedef typename core::term_traits<Term>::term_type term_type;
    typedef typename core::term_traits<Term>::variable_type variable_type;

    control_flow_simplifying_rewriter(const DataRewriter& rewriter)
      : m_rewriter(rewriter)
    {}

    term_type operator()(const term_type& x) const
    {
      control_flow_simplify_quantifier_builder<Term, DataRewriter> r(m_rewriter);
      return r(x);
    }

    template <typename SubstitutionFunction>
    term_type operator()(const term_type& x, SubstitutionFunction sigma) const
    {
      control_flow_simplify_quantifier_builder<Term, DataRewriter, SubstitutionFunction> r(m_rewriter);
      return r(x, sigma);
    }
};

/// \brief Algorithm class for the control_flow algorithm
class pbes_control_flow_algorithm
{
  public:
    struct control_flow_vertex;

    // edge of the control flow graph
    struct control_flow_edge
    {
      control_flow_vertex* source;
      control_flow_vertex* target;
      propositional_variable_instantiation label;

      control_flow_edge(control_flow_vertex* source_,
                        control_flow_vertex* target_,
                        const propositional_variable_instantiation& label_
                       )
       : source(source_),
         target(target_),
         label(label_)
       {}
    };

    // vertex of the control flow graph
    struct control_flow_vertex
    {
      propositional_variable_instantiation X;
      std::vector<control_flow_edge> incoming_edges;
      std::vector<control_flow_edge> outgoing_edges;
      pbes_expression guard;
      std::set<data::variable> marking;

      control_flow_vertex(const propositional_variable_instantiation& X_, pbes_expression guard_ = true_())
        : X(X_), guard(guard_)
      {}

      std::string print() const
      {
        std::ostringstream out;
        out << pbes_system::pp(X);
        out << " edges:";
        for (std::vector<control_flow_edge>::const_iterator i = outgoing_edges.begin(); i != outgoing_edges.end(); ++i)
        {
          out << " " << pbes_system::pp(i->source->X);
        }
        return out.str();
      }
    };

    struct control_flow_substitution
    {
      std::map<std::size_t, data::data_expression> values;

      propositional_variable_instantiation operator()(const propositional_variable_instantiation& x) const
      {
        data::data_expression_vector e = atermpp::convert<data::data_expression_vector>(x.parameters());
        for (std::map<std::size_t, data::data_expression>::const_iterator i = values.begin(); i != values.end(); ++i)
        {
          e[i->first] = i->second;
        }
        return propositional_variable_instantiation(x.name(), atermpp::convert<data::data_expression_list>(e));
      }
    };

    // simplify and rewrite the expression x
    pbes_expression simplify(const pbes_expression& x) const
    {
      data::detail::simplify_rewriter r;
      control_flow_simplifying_rewriter<pbes_expression, data::detail::simplify_rewriter> R(r);
      return R(x);
    }

    // simplify and rewrite the guards of the pbes p
    void simplify(pfnf_pbes& p) const
    {
      std::vector<pfnf_equation>& equations = p.equations();
      for (std::vector<pfnf_equation>::iterator k = equations.begin(); k != equations.end(); ++k)
      {
        simplify(k->h());
        std::vector<pfnf_implication>& implications = k->implications();
        for (std::vector<pfnf_implication>::iterator i = implications.begin(); i != implications.end(); ++i)
        {
          simplify(i->g());
        }
      }
    }

    pbes_control_flow_algorithm(const pbes<>& p)
      : m_pbes(p)
    {
      simplify(m_pbes);
    }

  protected:
    typedef atermpp::map<propositional_variable_instantiation, control_flow_vertex>::iterator vertex_iterator;

    // vertices of the control flow graph
    atermpp::map<propositional_variable_instantiation, control_flow_vertex> m_control_vertices;

    // the pbes that is considered
    pfnf_pbes m_pbes;

    // the control flow parameters
    std::map<core::identifier_string, std::vector<bool> > m_is_control_flow;

    propositional_variable find_propvar(const pbes<>& p, const core::identifier_string& X) const
    {
      const atermpp::vector<pbes_equation>& equations = p.equations();
      for (atermpp::vector<pbes_equation>::const_iterator i = equations.begin(); i != equations.end(); ++i)
      {
        if (i->variable().name() == X)
        {
          return i->variable();
        }
      }
      throw mcrl2::runtime_error("find_propvar failed!");
      return propositional_variable();
    }

//    // extract the propositional variable instantiations from an expression of the form g => \/_j in J . X_j(e_j)
//    std::vector<propositional_variable_instantiation> find_propositional_variables(const pbes_expression& x) const
//    {
//      std::vector<pbes_expression> v;
//      pbes_expression y = x;
//      if (is_imp(y))
//      {
//        y = imp(y).right();
//      }
//      split_or(y, v);
//
//      std::vector<propositional_variable_instantiation> result;
//      for (std::vector<pbes_expression>::iterator i = v.begin(); i != v.end(); ++i)
//      {
//        if (is_propositional_variable_instantiation(*i))
//        {
//          result.push_back(*i);
//        }
//      }
//      return result;
//    }

    void print_control_flow_parameters()
    {
      std::cout << "--- control flow parameters ---" << std::endl;
      const std::vector<pfnf_equation>& equations = m_pbes.equations();
      for (std::vector<pfnf_equation>::const_iterator k = equations.begin(); k != equations.end(); ++k)
      {
        propositional_variable X = k->variable();
        const std::vector<data::variable>& d_X = k->parameters();
        const std::vector<bool>& cf = m_is_control_flow[X.name()];

        std::cout << core::pp(X.name()) << " ";
        for (std::size_t i = 0; i < cf.size(); ++i)
        {
          if (cf[i])
          {
            std::cout << data::pp(d_X[i]) << " ";
          }
        }
        std::cout << std::endl;
      }
    }

    void compute_control_flow_parameters()
    {
      const std::vector<pfnf_equation>& equations = m_pbes.equations();
      std::map<core::identifier_string, std::vector<data::variable> > V;

      // initialize all control flow parameters to true
      // initalize V_km to the empty set
      for (std::vector<pfnf_equation>::const_iterator k = equations.begin(); k != equations.end(); ++k)
      {
        propositional_variable X = k->variable();
        const std::vector<data::variable>& d_X = k->parameters();
        m_is_control_flow[X.name()] = std::vector<bool>(d_X.size(), true);
        V[X.name()] = std::vector<data::variable>(d_X.size(), data::variable());
      }

      // pass 1
      for (std::vector<pfnf_equation>::const_iterator k = equations.begin(); k != equations.end(); ++k)
      {
        propositional_variable X = k->variable();
        const std::vector<data::variable>& d_X = k->parameters();
        const std::vector<pfnf_implication>& implications = k->implications();
        for (std::vector<pfnf_implication>::const_iterator i = implications.begin(); i != implications.end(); ++i)
        {
          const std::vector<propositional_variable_instantiation>& propvars = i->variables();
          for (std::vector<propositional_variable_instantiation>::const_iterator j = propvars.begin(); j != propvars.end(); ++j)
          {
            const propositional_variable_instantiation& Xij = *j;
            data::data_expression_list d = Xij.parameters();
            std::size_t index = 0;
            for (data::data_expression_list::const_iterator q = d.begin(); q != d.end(); ++q, ++index)
            {
              if (data::is_variable(*q))
              {
                std::vector<data::variable>::const_iterator found = std::find(d_X.begin(), d_X.end(), *q);
                if (found != d_X.end())
                {
                  if (V[Xij.name()][index] == data::variable())
                  {
                    V[Xij.name()][index] = *q;
                  }
                  else
                  {
                    m_is_control_flow[Xij.name()][index] = false;
                  }
                  std::cout << "pass 1: equation " << pbes_system::pp(X) << " variable " << pbes_system::pp(Xij) << " " << index << std::endl;
                }
              }
            }
          }
        }
      }

      // pass 2
      for (std::vector<pfnf_equation>::const_iterator k = equations.begin(); k != equations.end(); ++k)
      {
        propositional_variable X = k->variable();
        const std::vector<data::variable>& d_X = k->parameters();
        const std::vector<pfnf_implication>& implications = k->implications();
        for (std::vector<pfnf_implication>::const_iterator i = implications.begin(); i != implications.end(); ++i)
        {
          const std::vector<propositional_variable_instantiation>& propvars = i->variables();
          for (std::vector<propositional_variable_instantiation>::const_iterator j = propvars.begin(); j != propvars.end(); ++j)
          {
            const propositional_variable_instantiation& Xij = *j;
            data::data_expression_list d = Xij.parameters();
            std::size_t index = 0;
            for (data::data_expression_list::const_iterator q = d.begin(); q != d.end(); ++q, ++index)
            {
              if (is_constant(*q))
              {
                continue;
              }
              else if (data::is_variable(*q))
              {
                std::vector<data::variable>::const_iterator found = std::find(d_X.begin(), d_X.end(), *q);
                if (found == d_X.end())
                {
                  m_is_control_flow[Xij.name()][index] = false;
                  std::cout << "equation " << pbes_system::pp(X) << " variable " << pbes_system::pp(Xij) << " " << index << std::endl;
                }
                else
                {
                  if (X.name() == Xij.name() && (found != d_X.begin() + index))
                  {
                    m_is_control_flow[Xij.name()][index] = false;
                    std::cout << "equation " << pbes_system::pp(X) << " variable " << pbes_system::pp(Xij) << " " << index << std::endl;
                  }
                }
              }
              else
              {
                m_is_control_flow[Xij.name()][index] = false;
                std::cout << "equation " << pbes_system::pp(X) << " variable " << pbes_system::pp(Xij) << " " << index << std::endl;
              }
            }
          }
        }
      }
    }

    const std::vector<bool>& control_flow_values(const core::identifier_string& X) const
    {
      std::map<core::identifier_string, std::vector<bool> >::const_iterator i = m_is_control_flow.find(X);
      assert (i != m_is_control_flow.end());
      return i->second;
    }

    std::vector<data::variable> control_flow_parameters(const propositional_variable& X) const
    {
      std::vector<data::variable> result;
      const std::vector<bool>& b = control_flow_values(X.name());
      const data::variable_list& d = X.parameters();
      std::size_t index = 0;
      for (data::variable_list::const_iterator i = d.begin(); i != d.end(); ++i, index++)
      {
        if (b[index])
        {
          result.push_back(*i);
        }
      }
      return result;
    }

    // removes parameter values that do not correspond to a control flow parameter
    propositional_variable_instantiation project(const propositional_variable_instantiation& x) const
    {
      core::identifier_string X = x.name();
      data::data_expression_list d_X = x.parameters();
      const std::vector<bool>& b = control_flow_values(X);
      std::size_t index = 0;
      std::vector<data::data_expression> d;
      for (data::data_expression_list::iterator i = d_X.begin(); i != d_X.end(); ++i, index++)
      {
        if (b[index])
        {
          d.push_back(*i);
        }
      }
      return propositional_variable_instantiation(X, data::data_expression_list(d.begin(), d.end()));
    }

    // removes parameter values that do not correspond to a control flow parameter
    propositional_variable project_variable(const propositional_variable& x) const
    {
      core::identifier_string X = x.name();
      data::variable_list d_X = x.parameters();
      const std::vector<bool>& b = control_flow_values(X);
      std::size_t index = 0;
      std::vector<data::variable> d;
      for (data::variable_list::iterator i = d_X.begin(); i != d_X.end(); ++i, index++)
      {
        if (b[index])
        {
          d.push_back(*i);
        }
      }
      return propositional_variable(X, data::variable_list(d.begin(), d.end()));
    }

    // x is a projected value
    // \pre x is not present in m_control_vertices
    vertex_iterator insert_control_flow_vertex(const propositional_variable_instantiation& X, pbes_expression guard = true_())
    {
      std::pair<vertex_iterator, bool> p = m_control_vertices.insert(std::make_pair(X, control_flow_vertex(X, guard)));
      assert(p.second);
      return p.first;
    }

    template <typename Substitution>
    propositional_variable_instantiation apply_substitution(const propositional_variable_instantiation& X, Substitution sigma) const
    {
      return propositional_variable_instantiation(X.name(), data::replace_free_variables(X.parameters(), sigma));
    }

    void compute_control_flow_graph()
    {
      compute_control_flow_parameters();

      std::set<control_flow_vertex*> todo;

      // handle the initial state
      propositional_variable_instantiation Xinit = project(m_pbes.initial_state());
      vertex_iterator i = insert_control_flow_vertex(Xinit);
      todo.insert(&(i->second));

      while (!todo.empty())
      {
        std::set<control_flow_vertex*>::iterator i = todo.begin();
        todo.erase(i);
        control_flow_vertex& v = **i;
        control_flow_vertex* source = &v;
        std::cout << "selected todo element " << pbes_system::pp(v.X) << std::endl;

        const pfnf_equation& eqn = *find_equation(m_pbes, v.X.name());
        propositional_variable X = project_variable(eqn.variable());
        data::variable_list d = X.parameters();
        data::data_expression_list e = v.X.parameters();
        data::sequence_sequence_substitution<data::variable_list, data::data_expression_list> sigma(d, e);

        const std::vector<pfnf_implication>& implications = eqn.implications();
        for (std::vector<pfnf_implication>::const_iterator i = implications.begin(); i != implications.end(); ++i)
        {
          const std::vector<propositional_variable_instantiation>& propvars = i->variables();
          pbes_expression guard = and_(eqn.h(), i->g());
          for (std::vector<propositional_variable_instantiation>::const_iterator j = propvars.begin(); j != propvars.end(); ++j)
          {
            propositional_variable_instantiation Xij = project(*j);
            propositional_variable_instantiation Y = apply_substitution(Xij, sigma);
            propositional_variable_instantiation label = Xij;
            vertex_iterator q = m_control_vertices.find(Y);
            if (q == m_control_vertices.end())
            {
              // vertex Y does not yet exist
              std::cout << "discovered " << pbes_system::pp(Y) << std::endl;
              vertex_iterator k = insert_control_flow_vertex(Y, guard);
              todo.insert(&(k->second));
              std::cout << "added todo element " << pbes_system::pp(k->first) << std::endl;
              control_flow_vertex* target = &(k->second);
              v.outgoing_edges.push_back(control_flow_edge(source, target, label));
            }
            else
            {
              control_flow_vertex* target = &(q->second);
              v.outgoing_edges.push_back(control_flow_edge(source, target, label));
            }
          }
        }
      }

      // add incoming edges
      for (atermpp::map<propositional_variable_instantiation, control_flow_vertex>::iterator i = m_control_vertices.begin(); i != m_control_vertices.end(); ++i)
      {
        control_flow_vertex& v = i->second;
        for (std::vector<control_flow_edge>::iterator j = v.outgoing_edges.begin(); j != v.outgoing_edges.end(); ++j)
        {
          control_flow_edge& e = *j;
          e.target->incoming_edges.push_back(e);
        }
      }
    }

    void print_control_flow_graph() const
    {
      std::cout << "--- control flow graph ---" << std::endl;
      for (atermpp::map<propositional_variable_instantiation, control_flow_vertex>::const_iterator i = m_control_vertices.begin(); i != m_control_vertices.end(); ++i)
      {
        std::cout << "vertex " << i->second.print() << std::endl;
      }
    }

    void compute_control_flow_marking()
    {
      // initialization
      for (atermpp::map<propositional_variable_instantiation, control_flow_vertex>::iterator i = m_control_vertices.begin(); i != m_control_vertices.end(); ++i)
      {
        control_flow_vertex& v = i->second;
        std::set<data::variable> fv = pbes_system::find_free_variables(v.guard);

        const pfnf_equation& eqn = *find_equation(m_pbes, v.X.name());
    	  propositional_variable X = eqn.variable();
    	  const std::vector<data::variable>& d_X = eqn.parameters();

        std::vector<data::variable> to_be_removed = control_flow_parameters(X);
        for (std::set<data::variable>::iterator j = fv.begin(); j != fv.end(); ++j)
        {
        	if (std::find(d_X.begin(), d_X.end(), *j) == d_X.end())
          {
          	to_be_removed.push_back(*j);
          }
        }
        for (std::vector<data::variable>::iterator k = to_be_removed.begin(); k != to_be_removed.end(); ++k)
        {
        	fv.erase(*k);
        }
        v.marking = fv;
      }

      // backwards reachability algorithm
      std::set<control_flow_vertex*> todo;
      for (atermpp::map<propositional_variable_instantiation, control_flow_vertex>::iterator i = m_control_vertices.begin(); i != m_control_vertices.end(); ++i)
      {
        control_flow_vertex& v = i->second;
        todo.insert(&v);
      }
      while (!todo.empty())
      {
        std::set<control_flow_vertex*>::iterator i = todo.begin();
        todo.erase(i);
        control_flow_vertex& v = **i;
        std::cout << "selected todo element " << pbes_system::pp(v.X) << std::endl;

        for (std::vector<control_flow_edge>::iterator i = v.incoming_edges.begin(); i != v.incoming_edges.end(); ++i)
        {
          control_flow_vertex& u = *(i->source);
          std::size_t sz = u.marking.size();
          const propositional_variable_instantiation& Xij = i->label;
          std::set<data::variable> fv = pbes_system::find_free_variables(Xij);
        }
      }
    }

  public:

    /// \brief Runs the control_flow algorithm
    void run()
    {
      control_flow_influence_graph_algorithm ialgo(m_pbes);
      ialgo.run();

      control_flow_source_dest_algorithm sdalgo(m_pbes);
      sdalgo.compute_source_destination();
      sdalgo.print_source_destination();
      sdalgo.rewrite_propositional_variables();
      // N.B. This modifies m_pbes. It is needed as a precondition for the
      // function compute_control_flow_parameters().

      compute_control_flow_graph();
      print_control_flow_parameters();
      print_control_flow_graph();
      compute_control_flow_marking();
    }
};

} // namespace detail

} // namespace pbes_system

} // namespace mcrl2

#endif // MCRL2_PBES_DETAIL_CONTROL_FLOW_H
