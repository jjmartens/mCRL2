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

    block_map[0] = b0;
    //initialize worklist
    worklist.push(0);

    // Mark dirty bottom states
    for (state_type s : b0.states) {
      if (sil_suc[s].empty()) {
        mark_dirty(s);
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
    return get_eq_class(s) == get_eq_class(t);
  }

private:
  typedef std::size_t state_type;
  typedef std::size_t label_type;
  typedef std::size_t block_type;
  typedef std::tuple<block_type, label_type, block_type> observation_type;
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
  };

  block_type* blocks;
  std::map<block_type, block> block_map;
  std::set<state_type> frontier;
 
  //state_type* loc2state;
  //state_type* state2loc;
  std::set<state_type>* sil_pred;
  std::set<state_type>* sil_suc;
  std::map<state_type, std::size_t> state2numdirtysuc;
  //Implement this later.
  //std::size_t* state2silentout;
  //std::size_t* state2silentin;

  std::queue<block_type> worklist;

  void mark_dirty_backwards_closure(state_type s)
  { 
    for (auto spre : sil_pred[s])
    {
      state2numdirtysuc[spre] += 1;
      if (!block_map[blocks[spre]].dirty_states.insert(spre).second)
      {
        //The state was already dirty, so we remove it from the frontier
        frontier.erase(spre);
      }
      else
      {
        //The state was not dirty, so we compute reverse closure.
        mark_dirty_backwards_closure(spre);
      }
    }
  }

  void mark_dirty(state_type s)
  {
    //Add to frontier if no silent marked state
    block* B = &block_map[blocks[s]];
    if (B->dirty_states.insert(s).second)
    {
      //The state was not dirty, so we add it to the frontier and compute reverse closure.
      frontier.insert(s);
      mark_dirty_backwards_closure(s);
    }
  }

  //Signature of a state
  void sig(const state_type& s, signature_type& retsignature, std::map<state_type, signature_type>& sigs)
  {
    for (auto t : suc[s])
    {
      if (!is_tau(t.first)) {
        retsignature.insert(std::make_tuple(blocks[s], t.first, blocks[t.second]));
      }
      else
      {
        // Union with signature of t if signature is known
        if (sigs.find(t.second) != sigs.end())
        {
          retsignature.insert(sigs[t.second].begin(), sigs[t.second].end());
        }
        if (blocks[t.second] != blocks[s])
        {
          // If not silent add observation
          retsignature.insert(std::make_tuple(blocks[s], t.first, blocks[t.second]));
        }
      }
    }
  }

  //Split block based on signature
  void split()
  {
    if (frontier.empty())
    {
      //Block is clean (should not happen)
      mCRL2log(mcrl2::log::debug) << "Partition clean!? should not happen." << std::endl;
      return ;
    }
#ifdef UORDERED
    std::unordered_map<signature_type, block_type, sigHash> sig2block;
#else  
    std::map<signature_type, block_type> sig2block;
#endif
    mCRL2log(mcrl2::log::info) << "Start splitting" << std::endl;
    signature_type signature;
    // TODO: This might hit complexity, maybe we can improve this by juggling references to correct signatures?
    std::map<state_type, signature_type> sigs;
    std::set<block_type> blocks_to_split;
    while(!frontier.empty())
    {
      state_type s = *(frontier.begin());
      frontier.erase(s);
      signature.clear();
      sig(s, signature, sigs);
      sigs[s] = signature;
      block* B = &block_map[blocks[s]];
      B->dirty_states.erase(s);
      if (B->dirty_states.empty())
      {
        blocks_to_split.insert(blocks[s]);
      }
      for (auto spre: sil_pred[s])
      {
        if (block_map[blocks[spre]].dirty_states.find(spre) != block_map[blocks[spre]].dirty_states.end())
        {
          state2numdirtysuc[spre] -= 1;
          if (state2numdirtysuc[spre] == 0)
          {
            frontier.insert(spre);
          }
        }
      }
    }

    for (auto b : blocks_to_split)
    {
      block* B = &block_map[b];
      // Block is dirty
      std::map<signature_type, block_type> sig2block;
      std::map<state_type, block_type> state2block;
      std::map<block_type, std::size_t> block2size;

      for (auto s : B->states)
      {
        if (sigs.find(s) == sigs.end())
        {
          // Clean state stays in old block count in 0;
          block2size[0] += 1;
          state2block[s] = 0;
        }
        else
        {
          if (sig2block.find(sigs[s]) == sig2block.end())
          {
            block_type newblock_id = block_map.size();
            block2size[newblock_id] = 0;
            block newblock;
            newblock.states = std::set<state_type>();
            newblock.dirty_states = std::set<state_type>();
            block_map[newblock_id] = newblock;
            sig2block[sigs[s]] = newblock_id;
          }
          state2block[s] = sig2block[sigs[s]];
          block2size[state2block[s]] += 1;
        }
      }
      // Split block
      // argmax block2size
      block_type maxblock = 0;
      std::size_t maxsize = block2size[0];
      for (auto b : block2size)
      {
        if (b.second > maxsize)
        {
          maxsize = b.second;
          maxblock = b.first;
        }
      }
      if (block2size[0] == 0)
      {
        // No clean states, so we create 1 less block.
        block_map.erase(block_map.size() -1);
      }
      
      std::set<state_type> states = B->states;
      for (auto s : states)
      {
        if (state2block[s] != maxblock)
        {
          block_type nbid = (state2block[s] == 0) ? maxblock: state2block[s];
          nbid = (block2size[0] == 0 && nbid > maxblock) ? nbid - 1 : nbid;
          block_map[nbid].states.insert(s);
          B->states.erase(s);
          blocks[s] = nbid;
        }
      }
    }
    return;
  }

  //Refine based on sigs
  void refine()
  {
    int iter = 0;
    int new_blocks = 0;
    int old_blocks = 0;
    mCRL2log(mcrl2::log::info) << "Start refinement" << std::endl;
    while (!frontier.empty())
    {
      block_type old_blocks = block_map.size();
      split();
      block_type new_blocks = block_map.size(); 
      for (block_type b = old_blocks; b < new_blocks; ++b)
      {
        for (auto s : block_map[b].states)
        {
          for (auto t : pred[s])
          {
            if (!is_tau(t.first) or blocks[t.second] != b)
            {
              mark_dirty(t.second);
            }
          }
        }
      }
      mCRL2log(mcrl2::log::info) << "New frontier starts with size: \"" << frontier.size() << "\"" << std::endl;
    }
    mCRL2log(mcrl2::log::info) << "Done total blocks: \"" << block_map.size() << "\"" << std::endl;
  }
};
}
}
}
#endif

