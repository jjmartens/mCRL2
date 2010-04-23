// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/fdr/process.h
/// \brief add your file description here.

#ifndef MCRL2_FDR_PROCESS_H
#define MCRL2_FDR_PROCESS_H

#include "mcrl2/fdr/term_include_files.h"
#include "mcrl2/fdr/dotted_expression.h"
#include "mcrl2/fdr/field.h"
#include "mcrl2/fdr/linkpar.h"
#include "mcrl2/fdr/set_expression.h"

namespace mcrl2 {

namespace fdr {

//--- start generated process expression class declarations ---//
/// \brief class process
class process: public atermpp::aterm_appl
{
  public:
    /// \brief Default constructor.
    process()
      : atermpp::aterm_appl(fdr::detail::constructProc())
    {}

    /// \brief Constructor.
    /// \param term A term
    process(atermpp::aterm_appl term)
      : atermpp::aterm_appl(term)
    {
      assert(fdr::detail::check_rule_Proc(m_term));
    }
};

/// \brief list of processs
typedef atermpp::term_list<process> process_list;

/// \brief vector of processs
typedef atermpp::vector<process>    process_vector;

/// \brief A stop
class stop: public process
{
  public:
    /// \brief Default constructor.
    stop();

    /// \brief Constructor.
    /// \param term A term
    stop(atermpp::aterm_appl term);
};

/// \brief A skip
class skip: public process
{
  public:
    /// \brief Default constructor.
    skip();

    /// \brief Constructor.
    /// \param term A term
    skip(atermpp::aterm_appl term);
};

/// \brief A chaos
class chaos: public process
{
  public:
    /// \brief Default constructor.
    chaos();

    /// \brief Constructor.
    /// \param term A term
    chaos(atermpp::aterm_appl term);

    /// \brief Constructor.
    chaos(const set_expression& set);

    set_expression set() const;
};

/// \brief A prefix
class prefix: public process
{
  public:
    /// \brief Default constructor.
    prefix();

    /// \brief Constructor.
    /// \param term A term
    prefix(atermpp::aterm_appl term);

    /// \brief Constructor.
    prefix(const dotted_expression& dotted, const field_list& fields, const process& proc);

    dotted_expression dotted() const;

    field_list fields() const;

    process proc() const;
};

/// \brief An external choice
class externalchoice: public process
{
  public:
    /// \brief Default constructor.
    externalchoice();

    /// \brief Constructor.
    /// \param term A term
    externalchoice(atermpp::aterm_appl term);

    /// \brief Constructor.
    externalchoice(const process& left, const process& right);

    process left() const;

    process right() const;
};

/// \brief An internal choice
class internalchoice: public process
{
  public:
    /// \brief Default constructor.
    internalchoice();

    /// \brief Constructor.
    /// \param term A term
    internalchoice(atermpp::aterm_appl term);

    /// \brief Constructor.
    internalchoice(const process& left, const process& right);

    process left() const;

    process right() const;
};

/// \brief A sequential composition
class sequentialcomposition: public process
{
  public:
    /// \brief Default constructor.
    sequentialcomposition();

    /// \brief Constructor.
    /// \param term A term
    sequentialcomposition(atermpp::aterm_appl term);

    /// \brief Constructor.
    sequentialcomposition(const process& left, const process& right);

    process left() const;

    process right() const;
};

/// \brief An interrupt
class interrupt: public process
{
  public:
    /// \brief Default constructor.
    interrupt();

    /// \brief Constructor.
    /// \param term A term
    interrupt(atermpp::aterm_appl term);

    /// \brief Constructor.
    interrupt(const process& left, const process& right);

    process left() const;

    process right() const;
};

/// \brief An hiding
class hiding: public process
{
  public:
    /// \brief Default constructor.
    hiding();

    /// \brief Constructor.
    /// \param term A term
    hiding(atermpp::aterm_appl term);

    /// \brief Constructor.
    hiding(const process& proc, const set_expression& set);

    process proc() const;

    set_expression set() const;
};

/// \brief An interleave
class interleave: public process
{
  public:
    /// \brief Default constructor.
    interleave();

    /// \brief Constructor.
    /// \param term A term
    interleave(atermpp::aterm_appl term);

    /// \brief Constructor.
    interleave(const process& left, const process& right);

    process left() const;

    process right() const;
};

/// \brief A sharing
class sharing: public process
{
  public:
    /// \brief Default constructor.
    sharing();

    /// \brief Constructor.
    /// \param term A term
    sharing(atermpp::aterm_appl term);

    /// \brief Constructor.
    sharing(const process& left, const process& right, const set_expression& set);

    process left() const;

    process right() const;

    set_expression set() const;
};

/// \brief An alpha parallel
class alphaparallel: public process
{
  public:
    /// \brief Default constructor.
    alphaparallel();

    /// \brief Constructor.
    /// \param term A term
    alphaparallel(atermpp::aterm_appl term);

    /// \brief Constructor.
    alphaparallel(const process& left, const process& right, const set_expression& left_set, const set_expression& right_set);

    process left() const;

    process right() const;

    set_expression left_set() const;

    set_expression right_set() const;
};

/// \brief A replicated external choice
class repexternalchoice: public process
{
  public:
    /// \brief Default constructor.
    repexternalchoice();

    /// \brief Constructor.
    /// \param term A term
    repexternalchoice(atermpp::aterm_appl term);

    /// \brief Constructor.
    repexternalchoice(const setgen& gen, const process& proc);

    setgen gen() const;

    process proc() const;
};

/// \brief A replicated internal choice
class repinternalchoice: public process
{
  public:
    /// \brief Default constructor.
    repinternalchoice();

    /// \brief Constructor.
    /// \param term A term
    repinternalchoice(atermpp::aterm_appl term);

    /// \brief Constructor.
    repinternalchoice(const setgen& gen, const process& proc);

    setgen gen() const;

    process proc() const;
};

/// \brief A replicated sequential composition
class repsequentialcomposition: public process
{
  public:
    /// \brief Default constructor.
    repsequentialcomposition();

    /// \brief Constructor.
    /// \param term A term
    repsequentialcomposition(atermpp::aterm_appl term);

    /// \brief Constructor.
    repsequentialcomposition(const seqgen& gen, const process& proc);

    seqgen gen() const;

    process proc() const;
};

/// \brief A replicated interleave
class repinterleave: public process
{
  public:
    /// \brief Default constructor.
    repinterleave();

    /// \brief Constructor.
    /// \param term A term
    repinterleave(atermpp::aterm_appl term);

    /// \brief Constructor.
    repinterleave(const setgen& gen, const process& proc);

    setgen gen() const;

    process proc() const;
};

/// \brief A replicated sharing
class repsharing: public process
{
  public:
    /// \brief Default constructor.
    repsharing();

    /// \brief Constructor.
    /// \param term A term
    repsharing(atermpp::aterm_appl term);

    /// \brief Constructor.
    repsharing(const setgen& gen, const process& proc, const set_expression& set);

    setgen gen() const;

    process proc() const;

    set_expression set() const;
};

/// \brief A replicated alpha parallel
class repalphaparallel: public process
{
  public:
    /// \brief Default constructor.
    repalphaparallel();

    /// \brief Constructor.
    /// \param term A term
    repalphaparallel(atermpp::aterm_appl term);

    /// \brief Constructor.
    repalphaparallel(const setgen& gen, const process& proc, const set_expression& set);

    setgen gen() const;

    process proc() const;

    set_expression set() const;
};

/// \brief An untimed time-out
class untimedtimeout: public process
{
  public:
    /// \brief Default constructor.
    untimedtimeout();

    /// \brief Constructor.
    /// \param term A term
    untimedtimeout(atermpp::aterm_appl term);

    /// \brief Constructor.
    untimedtimeout(const process& left, const process& right);

    process left() const;

    process right() const;
};

/// \brief A boolean guard
class boolguard: public process
{
  public:
    /// \brief Default constructor.
    boolguard();

    /// \brief Constructor.
    /// \param term A term
    boolguard(atermpp::aterm_appl term);

    /// \brief Constructor.
    boolguard(const boolean_expression& guard, const process& proc);

    boolean_expression guard() const;

    process proc() const;
};

/// \brief A linked parallel
class linkedparallel: public process
{
  public:
    /// \brief Default constructor.
    linkedparallel();

    /// \brief Constructor.
    /// \param term A term
    linkedparallel(atermpp::aterm_appl term);

    /// \brief Constructor.
    linkedparallel(const process& left, const process& right, const linkpar& linked);

    process left() const;

    process right() const;

    linkpar linked() const;
};

/// \brief A replicated linked parallel
class replinkedparallel: public process
{
  public:
    /// \brief Default constructor.
    replinkedparallel();

    /// \brief Constructor.
    /// \param term A term
    replinkedparallel(atermpp::aterm_appl term);

    /// \brief Constructor.
    replinkedparallel(const seqgen& gen, const process& proc, const linkpar& linked);

    seqgen gen() const;

    process proc() const;

    linkpar linked() const;
};
//--- end generated process expression class declarations ---//

//--- start generated is-functions ---//

    /// \brief Test for a stop expression
    /// \param t A term
    /// \return True if it is a stop expression
    inline
    bool is_stop(const process& t)
    {
      return fdr::detail::gsIsSTOP(t);
    }

    /// \brief Test for a skip expression
    /// \param t A term
    /// \return True if it is a skip expression
    inline
    bool is_skip(const process& t)
    {
      return fdr::detail::gsIsSKIP(t);
    }

    /// \brief Test for a chaos expression
    /// \param t A term
    /// \return True if it is a chaos expression
    inline
    bool is_chaos(const process& t)
    {
      return fdr::detail::gsIsCHAOS(t);
    }

    /// \brief Test for a prefix expression
    /// \param t A term
    /// \return True if it is a prefix expression
    inline
    bool is_prefix(const process& t)
    {
      return fdr::detail::gsIsPrefix(t);
    }

    /// \brief Test for a externalchoice expression
    /// \param t A term
    /// \return True if it is a externalchoice expression
    inline
    bool is_externalchoice(const process& t)
    {
      return fdr::detail::gsIsExternalChoice(t);
    }

    /// \brief Test for a internalchoice expression
    /// \param t A term
    /// \return True if it is a internalchoice expression
    inline
    bool is_internalchoice(const process& t)
    {
      return fdr::detail::gsIsInternalChoice(t);
    }

    /// \brief Test for a sequentialcomposition expression
    /// \param t A term
    /// \return True if it is a sequentialcomposition expression
    inline
    bool is_sequentialcomposition(const process& t)
    {
      return fdr::detail::gsIsSequentialComposition(t);
    }

    /// \brief Test for a interrupt expression
    /// \param t A term
    /// \return True if it is a interrupt expression
    inline
    bool is_interrupt(const process& t)
    {
      return fdr::detail::gsIsInterrupt(t);
    }

    /// \brief Test for a hiding expression
    /// \param t A term
    /// \return True if it is a hiding expression
    inline
    bool is_hiding(const process& t)
    {
      return fdr::detail::gsIsHiding(t);
    }

    /// \brief Test for a interleave expression
    /// \param t A term
    /// \return True if it is a interleave expression
    inline
    bool is_interleave(const process& t)
    {
      return fdr::detail::gsIsInterleave(t);
    }

    /// \brief Test for a sharing expression
    /// \param t A term
    /// \return True if it is a sharing expression
    inline
    bool is_sharing(const process& t)
    {
      return fdr::detail::gsIsSharing(t);
    }

    /// \brief Test for a alphaparallel expression
    /// \param t A term
    /// \return True if it is a alphaparallel expression
    inline
    bool is_alphaparallel(const process& t)
    {
      return fdr::detail::gsIsAlphaParallel(t);
    }

    /// \brief Test for a repexternalchoice expression
    /// \param t A term
    /// \return True if it is a repexternalchoice expression
    inline
    bool is_repexternalchoice(const process& t)
    {
      return fdr::detail::gsIsRepExternalChoice(t);
    }

    /// \brief Test for a repinternalchoice expression
    /// \param t A term
    /// \return True if it is a repinternalchoice expression
    inline
    bool is_repinternalchoice(const process& t)
    {
      return fdr::detail::gsIsRepInternalChoice(t);
    }

    /// \brief Test for a repsequentialcomposition expression
    /// \param t A term
    /// \return True if it is a repsequentialcomposition expression
    inline
    bool is_repsequentialcomposition(const process& t)
    {
      return fdr::detail::gsIsRepSequentialComposition(t);
    }

    /// \brief Test for a repinterleave expression
    /// \param t A term
    /// \return True if it is a repinterleave expression
    inline
    bool is_repinterleave(const process& t)
    {
      return fdr::detail::gsIsRepInterleave(t);
    }

    /// \brief Test for a repsharing expression
    /// \param t A term
    /// \return True if it is a repsharing expression
    inline
    bool is_repsharing(const process& t)
    {
      return fdr::detail::gsIsRepSharing(t);
    }

    /// \brief Test for a repalphaparallel expression
    /// \param t A term
    /// \return True if it is a repalphaparallel expression
    inline
    bool is_repalphaparallel(const process& t)
    {
      return fdr::detail::gsIsRepAlphaParallel(t);
    }

    /// \brief Test for a untimedtimeout expression
    /// \param t A term
    /// \return True if it is a untimedtimeout expression
    inline
    bool is_untimedtimeout(const process& t)
    {
      return fdr::detail::gsIsUntimedTimeOut(t);
    }

    /// \brief Test for a boolguard expression
    /// \param t A term
    /// \return True if it is a boolguard expression
    inline
    bool is_boolguard(const process& t)
    {
      return fdr::detail::gsIsBoolGuard(t);
    }

    /// \brief Test for a linkedparallel expression
    /// \param t A term
    /// \return True if it is a linkedparallel expression
    inline
    bool is_linkedparallel(const process& t)
    {
      return fdr::detail::gsIsLinkedParallel(t);
    }

    /// \brief Test for a replinkedparallel expression
    /// \param t A term
    /// \return True if it is a replinkedparallel expression
    inline
    bool is_replinkedparallel(const process& t)
    {
      return fdr::detail::gsIsRepLinkedParallel(t);
    }
//--- end generated is-functions ---//

} // namespace fdr

} // namespace mcrl2

#endif // MCRL2_FDR_PROCESS_H
