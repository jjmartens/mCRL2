// Author(s): Jan Martens
//
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
/// \file lts/detail/liblts_bisim_m.h
///
/// \brief Partition refinement for Branching bisimularity reduction.
/// Inspired by Jules Jacobs, Thorsten Wiﬂmann,  "Fast coalgebraic bisimilarity minimization." - POPL2023.
///
/// \details 
#ifndef _LIBLTS_BISIM_MARTENS
#define _LIBLTS_BISIM_MARTENS
#include <fstream>
#include "mcrl2/modal_formula/state_formula.h"
#include "mcrl2/lts/lts_utilities.h"
#include "mcrl2/lts/detail/liblts_scc.h"
#include "mcrl2/lts/detail/liblts_merge.h"
#include "mcrl2/lts/lts_aut.h"
#include "mcrl2/lts/lts_fsm.h"
#include "mcrl2/lts/lts_dot.h"

//#define UORDERED
namespace mcrl2
{
namespace lts
{
namespace detail
{
template < class LTS_TYPE>
class bisim_partitioner_martens
{
public:
  /** \brief Creates a bisimulation partitioner for an LTS.
    *  \details Based on the paper "Fast coalgebraic bisimilarity minimization." - Jacobs, Jules, and Thorsten Wiﬂmann. Proceedings of the ACM on Programming Languages 7.POPL (2023): 1514-1541.
    *  \warning Experimental.
    *  \param[in] l Reference to the LTS. */
  bisim_partitioner_martens(
    LTS_TYPE& l)
    : max_state_index(0),
    aut(l)
  {
    auto start = std::chrono::high_resolution_clock::now();

    const std::vector<transition>& trans = aut.get_transitions();

    //Initialize arrays for pred and suc, 
    pred = new std::vector<custom_transition_type>[aut.num_states()];
    suc = new std::vector<custom_transition_type>[aut.num_states()];

    //TODO optimization: Derive these by partitioning the transition array
    sil_pred = new std::set<state_type>[aut.num_states()];
    sil_suc = new std::set<state_type>[aut.num_states()];

    blocks = new block_type[aut.num_states()];
    std::vector<int> state2in = std::vector<int>(aut.num_states(), 0);
    std::vector<int> state2out = std::vector<int>(aut.num_states(), 0);
    mCRL2log(mcrl2::log::debug) << "start moving transitions " << std::endl;

    //Count transitions per state
    for (auto r = trans.begin(); r != trans.end(); r++) {
      state2in[(*r).to()] += 1;
      state2out[(*r).from()] += 1;
    }

    std::set<state_type> allstates;
    std::set<state_type> bottomstates;
    for (state_type s = 0; s < aut.num_states(); ++s)
    {
      pred[s] = std::vector<custom_transition_type>(state2in[s]);
      suc[s] = std::vector<custom_transition_type>(state2out[s]);
      sil_pred[s] = std::set<state_type>();
      sil_suc[s] = std::set<state_type>();
      allstates.insert(s);
    }

    for (auto r = trans.begin(); r != trans.end(); r++)
    {
      state2in[(*r).to()] -= 1;
      pred[(*r).to()][state2in[(*r).to()]] = std::make_pair((*r).label(), (*r).from());
      state2out[(*r).from()] -= 1;
      suc[(*r).from()][state2out[(*r).from()]] = std::make_pair((*r).label(), (*r).to());
      if (is_tau((*r).label())) {
        sil_pred[(*r).to()].insert((*r).from());
        sil_suc[(*r).from()].insert((*r).to());
      }
      mCRL2log(mcrl2::log::debug) << "transition " << (*r).from() << " " << (*r).label() << " " << (*r).to() << std::endl;
    }
    //Initialize block map
    worklist = std::queue<block_type>();
    block_map = std::map<block_type, block>();

    //Initialize blocks
    for (state_type i = 0; i < (unsigned int)aut.num_states(); ++i)
    {
      if (sil_suc[i].size() == 0)
      {
        bottomstates.insert(i);
      }
      blocks[i] = 0;
    }

    block b0;
    b0.states = allstates;
    b0.dirty_states = allstates;
    b0.frontier = bottomstates;

    block_map[0] = b0;
    //initialize worklist
    worklist.push(0);

    // Mark dirty bottom states
    for (state_type s : b0.states) {
      if (sil_suc[s].empty()) {
        b0.dirty_states.insert(s);
      }
    }

    // Iterate refinement 
    refine();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    mCRL2log(mcrl2::log::info) << "Done s:" << duration.count() << std::endl;
    //cleanup
    delete[] pred;
    delete[] suc;
    delete[] blocks;
    delete[] sil_pred;
    delete[] sil_suc;
  }

  /** \brief Destroys this partitioner. */
  ~bisim_partitioner_martens() = default;

  /** \brief Gives the number of bisimulation equivalence classes of the LTS.
   *  \return The number of bisimulation equivalence classes of the LTS.
   */
  std::size_t num_eq_classes() const
  {
    return block_map.size();
  }


  /** \brief Gives the bisimulation equivalence class number of a state.
   *  \param[in] s A state number.
   *  \return The number of the bisimulation equivalence class to which \e s belongs. */
  std::size_t get_eq_class(const std::size_t s) const
  {
    return blocks[s];
  }

  /** \brief Returns whether two states are in the same bisimulation equivalence class.
   *  \param[in] s A state number.
   *  \param[in] t A state number.
   *  \retval true if \e s and \e t are in the same bisimulation equivalence class;
   *  \retval false otherwise. */
  bool in_same_class(const std::size_t s, const std::size_t t) const
  {
    mCRL2log(mcrl2::log::debug) << "in_same_class " << s << " " << t << " " << get_eq_class(s) << " " << get_eq_class(t) << std::endl;
    return get_eq_class(s) == get_eq_class(t);
  }

private:
  typedef std::size_t state_type;
  typedef std::size_t label_type;
  typedef std::size_t block_type;
  typedef std::pair<label_type, block_type> observation_type;
  typedef std::pair<label_type, state_type> custom_transition_type;

  //Typedef for signature
#ifdef UORDERED
  typedef std::set<observation_type> signature_type;
#else
  typedef std::set<observation_type> signature_type;
#endif

  bool is_tau(label_type l)
  {
    return aut.is_tau(aut.apply_hidden_label_map(l));
  }

  state_type max_state_index;
  LTS_TYPE& aut;
  // Array of vectors of predeccessors and successors
  std::vector<custom_transition_type>* pred;
  std::vector<custom_transition_type>* suc;

  //Struct block with start, mid, end being pointers
  // [start, mid) is clean, [mid, end) is dirty
  // [mid, bottom) is dirty and bottom.
  struct block
  {
//    state_type start;
//   state_type mid;
//   state_type end;
    //std::unordered_set<label_type> unstable_labels;
    std::set<state_type> states;
    std::set<state_type> dirty_states;
    std::set<state_type> frontier; 
  };

  block_type* blocks;
  std::map<block_type, block> block_map;
  //state_type* loc2state;
  //state_type* state2loc;
  std::set<state_type>* sil_pred;
  std::set<state_type>* sil_suc;
  
  //Implement this later.
  //std::size_t* state2silentout;
  //std::size_t* state2silentin;

  std::queue<block_type> worklist;

  void mark_dirty_backwards_closure(state_type s)
  { 
    for (auto spre : sil_pred[s])
    {
      if (blocks[spre] == blocks[s])
      {
        if (!block_map[blocks[s]].dirty_states.insert(spre).second)
        {
          //The state was already dirty, so we remove it from the frontier
          block_map[blocks[s]].frontier.erase(spre);
          mark_dirty_backwards_closure(spre);
        }
      }
    }
  }

  void mark_dirty(state_type s)
  {
    //Add to frontier if no silent marked state
    block* Bid = &block_map[blocks[s]];
    if (Bid->dirty_states.insert(s).second)
    {
      //The state was not dirty, so we add it to the frontier and compute reverse closure.
      if (Bid->frontier.empty()) {
        //First dirty state, so we add it to the worklist
        worklist.push(blocks[s]);
      }
      Bid->frontier.insert(s);
      mark_dirty_backwards_closure(s);
    }
  }

  //Signature of a state
  void sig(const state_type& s, signature_type& retsignature, std::map<state_type, signature_type>& sigs, block_type curblock)
  {
    for (auto t : suc[s])
    {
      if (blocks[t.second] != curblock or !is_tau(t.first)) {
        mCRL2log(mcrl2::log::debug) << "inserting " << t.first << " " << t.second << " in sig of " << s <<  std::endl;
        retsignature.insert(std::make_pair(t.first, blocks[t.second]));
        auto sig = std::make_pair(t.first, blocks[t.second]);
        // Add observation of r to sig
        retsignature.insert(sig);
      }
      else {
        if (block_map[curblock].dirty_states.find(t.second) == block_map[curblock].dirty_states.end())
        {
          retsignature.insert(std::make_pair(t.first, blocks[t.second]));
          auto sig = std::make_pair(t.first, curblock);
          // Add observation of r to sig curblock encodes it is able to go to old signature.
          retsignature.insert(sig);
        }
        else
        {
          // If it is dirty, we should include the signature of the state t.
          retsignature.insert(sigs[t.second].begin(), sigs[t.second].end());
        }
      }
    }
  }

  //Split block based on signature
  int split(block_type Bid)
  {
    block* B = &block_map[Bid];
    if (B->frontier.empty())
    {
      //Block is clean (should not happen)
      mCRL2log(mcrl2::log::debug) << "Block clean but in worklist!? should not happen." << std::endl;
      return 0;
    }
#ifdef UORDERED
    std::unordered_map<signature_type, block_type, sigHash> sig2block;
#else  
    std::map<signature_type, block_type> sig2block;
#endif
    int j = 0;
    std::map<state_type, block_type> state2block;

    mCRL2log(mcrl2::log::debug) << "Computing signatures from here. " << Bid << ": " << B->states.size() << " of whchi dirty: "
                                << B->dirty_states.size() << std::endl;

    for (auto s : B->states)
    {
      mCRL2log(mcrl2::log::debug) << "\tstate: " << s << " is vies: " << (B->dirty_states.find(s) != B->dirty_states.end()) << std::endl;
    }
    //Add signature of one clean state
    signature_type signature;
    signature.insert(std::make_pair(0, Bid));
    sig2block[signature] = 0;
    j += 1;

    //TODO: This might hit complexity, maybe we can improve this by juggling references to correct signatures?
    std::map<state_type, signature_type> sigs;

    //Add signatures of dirty states
    std::set<state_type> done;
    while(!B->frontier.empty())
    {
      state_type s = *(B->frontier.begin());
      B->frontier.erase(s);
      signature.clear();

      sig(s, signature, sigs, Bid);
      mCRL2log(mcrl2::log::debug) << "sig comped" << s << std::endl;
      sigs[s] = signature;

      auto ret = sig2block.insert(std::make_pair(signature, j));
      if (ret.second) {
        j += 1;
      }

      done.insert(s);
      for (auto spre: sil_pred[s])
      {
        if (blocks[spre] == Bid)
        {
          for (auto t : sil_suc[spre])
          {
            if (blocks[t] != Bid or (B->dirty_states.find(t) != B->dirty_states.end() and done.find(t) == done.end()))
            {
              break;
            }
          }
          B->frontier.insert(spre);
        }
      }
      state2block[s] = (*ret.first).second;
      mCRL2log(mcrl2::log::debug) << "new block of state " << s << " is:" << state2block[s] << std::endl;
    }
   
    if (j == 1) {
      //Only one signature, no need to split
      mCRL2log(mcrl2::log::debug) << "no new signatures." << std::endl;
      return 0;
    }
    std::map<block_type, std::set<state_type>> block2sets;
    std::set<state_type> og_states = B->states;
    mCRL2log(mcrl2::log::debug) << "og states :" << og_states.size() << std::endl;
    for (auto s : state2block)
    {
      block2sets[s.second].insert(s.first);    
    }
    std::set<state_type> nondirty_states = B->states;
    // nondirty states
    for (auto s : B->dirty_states)
    {
      nondirty_states.erase(s);
    }

    block2sets[0] = nondirty_states;
    // Find biggest block (argmax) 
    std::size_t maxsize = nondirty_states.size();
    block_type maxblock = 0;
    for (auto b : block2sets)
    {
      if (b.second.size() > maxsize)
      {
        maxsize = b.second.size();
        maxblock = b.first;
      }
    }

    // New blocks for each non maxblock
    for (auto b : block2sets)
    {
      if (b.first != maxblock and b.second.size() != 0)
      {
        block_type newblock = block_map.size();
        block Bnew;
        Bnew.states = b.second;
        Bnew.dirty_states = std::set<state_type>();
        Bnew.frontier = std::set<state_type>();
        for (auto s : b.second)
        {
          blocks[s] = newblock;
        }
        block_map[newblock] = Bnew;
      }
    }
    // Biggest block will become the original block.

    B->states.clear();
    B->dirty_states.clear();
    for (auto s : block2sets[maxblock])
    {
      B->states.insert(s);
    }

    // Compute silent transitions 
    mCRL2log(mcrl2::log::debug) << "og states2 :" << og_states.size() << std::endl;
    for (auto s : og_states)
    {
      std::set<state_type> sucs = sil_suc[s];
      for (auto t : sucs)
      {
        if (blocks[t] != blocks[s])
        {
          sil_suc[s].erase(t);
          sil_pred[t].erase(s);
        }
      }
      if (sil_suc[s].empty() and !sucs.empty())
      {
        // New bottom state, make it dirty. (maybe we can do this on the fly(inductive signatures?).)
        mark_dirty(s);
      }
    }
    mCRL2log(mcrl2::log::debug) << "Splitted the block:" << Bid << " into #newblocks" << j << std::endl;
    return j;
}

  //Refine based on sigs
  void refine()
  {
    int iter = 0;
    int new_blocks = 0;
    int old_blocks = 0;
    mCRL2log(mcrl2::log::info) << "Start refinement" << std::endl;
    while (!worklist.empty())
    {
      iter += 1;
      block_type Bid = worklist.front();
      worklist.pop();
      
      // ideally this maybe should be a pointer??
      block* B = &block_map[Bid];

      //Count states
      if (B->frontier.empty())
      {
        //Block is clean (should not happen)
        mCRL2log(mcrl2::log::info) << "Block clean but in worklist!? should not happen." << std::endl;
      } else {
        mCRL2log(mcrl2::log::debug) << "start iter" << std::endl;
        old_blocks = block_map.size();
        int ret = split(Bid);
        mCRL2log(mcrl2::log::info) << "new blocks created = " << block_map.size() - old_blocks << std::endl;
        if (old_blocks != block_map.size())
        {
          //From all new blocks, mark backwards dirty. 
          for (block_type i = old_blocks; i < block_map.size(); ++i)
          {
            for (auto s : block_map[i].states)
            {
              for (auto trans: pred[s])
              {
                if (!is_tau(trans.first) or blocks[trans.second] != i)
                {
                  mCRL2log(mcrl2::log::debug) << "marking dirty" << std::endl;
                  mark_dirty(trans.second);
                } 
              }
            }
          }
        }  
      }
    }
    mCRL2log(mcrl2::log::info) << "Done total blocks: \"" << block_map.size() << "\"" << std::endl;
  }
};
}
}
}
#endif

