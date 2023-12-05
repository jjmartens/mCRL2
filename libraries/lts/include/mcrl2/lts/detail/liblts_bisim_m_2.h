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
/// \brief Partition refinement algorithm for guaruanteed minimal depth
/// counter-examples.
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
       //Initialize arrays for pred and suc, blocks and state2loc and loc2state
       pred = new std::vector<transition>[aut.num_states()];
       suc  = new std::vector<transition>[aut.num_states()];
       blocks = new block_type[aut.num_states()];
       state2loc = new state_type[aut.num_states()];
       loc2state = new state_type[aut.num_states()];
       for (state_type s = 0; s < aut.num_states(); ++s)
	   {
           pred[s] = std::vector<transition>();
           suc[s] = std::vector<transition>();
	   }


       for (std::vector<transition>::const_iterator r = trans.begin(); r != trans.end(); ++r)
       {
           pred[(*r).to()].push_back(*r);
           suc[(*r).from()].push_back(*r);
       }

        //Initialize block map
       worklist = std::queue<block_type>();
       block_map = std::map<block_type, block>();
       
       block b0 = block{0,0,aut.num_states()};
        for (state_type i = 0; i < aut.num_states(); ++i)
        {
            blocks[i] = 0; 
        }

        block_map.insert(std::pair(0, b0));
        //initialize worklist
        worklist.push(0);

        //Initialize state2loc and loc2state
        for(std::size_t i = 0; i < aut.num_states(); ++i)
		{
			state2loc[i] = i;
			loc2state[i] = i;
		}

        // Iterate refinement 
        refine();

        delete[] pred;
        delete[] suc;
        delete[] state2loc;
        delete[] loc2state;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        mCRL2log(mcrl2::log::info) << "Done s:" << duration.count() <<  std::endl;
    }

    /** \brief Destroys this partitioner. */
    ~bisim_partitioner_martens() = default;

    /** \brief Gives the number of bisimulation equivalence classes of the LTS.
     *  \return The number of bisimulation equivalence classes of the LTS.
     */
    std::size_t num_eq_classes() const
    {
        return max_state_index;
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
        mCRL2log(mcrl2::log::debug) << "in_same_class " << s<< " "<< t<< " " << get_eq_class(s) << " " << get_eq_class(t) << std::endl;
        return get_eq_class(s) == get_eq_class(t);
    }

private:
    typedef std::size_t state_type;
    typedef std::size_t label_type;
    typedef std::size_t block_type;
    typedef std::pair<label_type, block_type> observation_type;
    
    struct signaturehash {
        std::size_t operator()(const std::set<observation_type>& obsset) const
        {
            long long hash{ 0 };
            for (auto obs : obsset) {
                hash = hash ^ (std::hash<std::size_t>()(obs.first) ^ std::hash<std::size_t>()(obs.second));
            }
            return hash;
        }
    };

    size_t hash_pair(const std::pair<size_t, size_t>& p) const
	{
		return std::hash<size_t>{}(p.first) ^ std::hash<size_t>{}(p.second);
	}



    //Typedef for signature
    typedef std::unordered_set<observation_type, signaturehash> signature_type;
    
    state_type max_state_index;
    LTS_TYPE& aut;
    // Array of vectors of predeccessors and successors
    std::vector<transition>* pred; 
    std::vector<transition>* suc;
    
    //Struct block with start, mid, end beingpointers
    struct block
	{
		state_type start;
		state_type mid;
		state_type end;

	};


    block_type* blocks;
    std::map<block_type, block> block_map;
    state_type* loc2state;
    state_type* state2loc;
    std::queue<block_type> worklist;

    void mark_dirty(state_type s)
	{
        block_type Bid = blocks[s];
        block B = block_map[Bid];
        state_type loc = state2loc[s];
        if (loc < B.start || loc > B.end) {
            mCRL2log(mcrl2::log::info) << "KAPUT "<< loc << " " << B.start << " " << B.mid << " " << B.end << std::endl;
        }

        if (B.mid <= loc) {
            //Already dirty
            return;
        }
        if (B.mid == B.end) {
            //First dirty state
            worklist.push(Bid);
        }
        //Swap last clean state with s
        state_type new_id = B.mid - 1;
        state_type tmp = loc2state[new_id];
        block_map[Bid].mid = new_id;

        state2loc[tmp] = loc;
        loc2state[loc] = tmp;
        state2loc[s] = new_id;
        loc2state[new_id] = s;
	}
    
    //Signature of a state
    signature_type sig(state_type s)
    {
        signature_type signature;
		for (std::vector<transition>::iterator r = suc[s].begin(); r != suc[s].end(); r++)
		{
            // Add observation of r to sig
            signature.insert(std::pair<label_type, block_type>((*r).label(), blocks[(*r).to()]));
		}
		return signature;
    }

    //Split block based on signature
    std::vector<block_type> split(block_type Bid)
    {
		block B = block_map[Bid];
		if (B.mid == B.end)
		{
			//Block is clean (should not happen)
			return std::vector<block_type>();
		}
		std::unordered_map<signature_type, block_type> sig2block;
        int j = 0;
        block_type* state2block = new block_type[B.end - B.mid];

        //Add signature of one clean state
        if (B.mid != B.start) {
			state_type s = loc2state[B.start];
			signature_type signature = sig(s);
			sig2block.insert(std::pair<signature_type, block_type>(signature, j));
            state2block[B.mid] = j;
            j += 1;
		}
		//Add signatures of dirty states
		for (state_type s = B.mid; s < B.end; s++)
		{
			signature_type signature = sig(loc2state[s]);
            if (sig2block.find(signature) == sig2block.end())
            {
                //New signature
                sig2block.insert(std::pair<signature_type, block_type>(signature, j));
                j += 1;
            }
            state2block.insert(std::pair<state_type, block_type>(loc2state[s], sig2block[signature]));
		}
        if (j == 1) {
            //Only one signature, no need to split
            block_map[Bid].mid = B.end;
            return std::vector<block_type>();
        }

        int* Sizes = new int[j+1];
        for (int i = 0; i <= j; i++)
		{
			Sizes[i] = 0;
		}
        //Sizes[0] = B.mid - B.start;
        for (state_type s = B.mid; s < B.end; s++)
        {
            Sizes[state2block[loc2state[s]]] += 1;
        }

        //Create new blocks
        // argmax Sizes
        state_type max = 0;
        int max_index = 0;
        Sizes[0] += B.mid - B.start;

        for (int i = 0; i < j; i++)
        {
            if (Sizes[i] > max) {
                max = Sizes[i];
                max_index = i;
            }
        }
        Sizes[0] -= B.mid - B.start;

        //Prefix sum Sizes
        for (int i = 0; i < j; i++)
        {
            Sizes[i+1] += Sizes[i];
        }
        int num_dirty = B.end - B.mid;
        state_type* dirty = new state_type[num_dirty];

        /*for (state_type i = 0; i < num_dirty; i++) {
			dirty[i] = loc2state[B.mid + i];
		}*/

        std::copy(loc2state + B.mid, loc2state + B.end, dirty);
        //Reorder states
        //i + 1 >= B.mid + 1 is a hack because of unsigned ints
        for (state_type i = 0; i < num_dirty; i++) {
            state_type si = dirty[i];
            if(Sizes[state2block[si]] == 0) {
				//Impossible
                mCRL2log(mcrl2::log::info) << "Impossible" << si << " " << state2block[si] << std::endl;
				continue;
			}
            Sizes[state2block[si]] -= 1;
            int tmp = B.mid + Sizes[state2block[si]];
            loc2state[tmp] = si;
            state2loc[si] = tmp;
		}

        if (Sizes[0] != 0 || Sizes[j] != num_dirty) {
            mCRL2log(mcrl2::log::info) << "Assertion failed Sizes inccorrect" << std::endl;
        }
        // create new blocks
        std::vector<block_type> new_blocks = std::vector<block_type>();
        state_type old_start = B.mid;
        state_type old_end = B.end;
        //mCRL2log(mcrl2::log::debug) << "Bmid" << B.mid << std::endl;
        /*int num_err = 0;
        for (state_type locs = B.start; locs < B.mid; locs++) {
            if(sig2block[sig(loc2state[locs])] != 0) {
                num_err += 1;
                mCRL2log(mcrl2::log::debug) << locs << std::endl;
            }
        }*/
        //mCRL2log(mcrl2::log::debug) << "num_err:" << num_err << std::endl;

        for (int i = 0; i < j; i++)
        {
            state_type newstart = old_start + Sizes[i];
            state_type newend = old_start + Sizes[i+1];
            if (i == 0) {
                newstart = B.start;
            }

            assert(newstart >= old_start);
            assert(newend <= old_end);
            if (i == max_index) {
                B.start = newstart;
                B.mid = newend;
                B.end = newend;
                block_map[Bid] = B;
            }
            else {
                block_type newBid = block_map.size();
                block newB = block{ newstart, newend, newend };
                block_map.insert(std::pair<block_type, block>(newBid, newB));
                for (state_type locs = newB.start; locs < newB.end; locs++)
                {
                   blocks[loc2state[locs]] = newBid; 
                }
                new_blocks.push_back(newBid);
            }
		}

        delete[] Sizes;
        delete[] dirty;
        return new_blocks;
    }

    //Refine based on sigs
    void refine()
    {
        int iter = 0;
        std::vector<block_type> new_blocks;

		while (!worklist.empty())
		{
            iter += 1;
			block_type Bid = worklist.front();
			worklist.pop();
			block B = block_map[Bid];

            int num_dirtyeightysource = 0;
            int num_dirtyeightytarget = 0;
            //Count states

			if (B.mid == B.end and B.end > B.start + 1)
			{
				//Block is clean (should not happen)
                mCRL2log(mcrl2::log::info) << "Block clean but in worklist!? should not happen." << std::endl;
            }
            else {
                //new_blocks = split_complete(Bid);
                new_blocks = split(Bid);

                for (auto blockid : new_blocks)
                {
                    block newB = block_map[blockid];

                    std::vector<state_type> overlap;
                    for (state_type s = newB.start; s < newB.end; s++) {
                        for (auto  r : pred[loc2state[s]]) {
                            if(blocks[r.from()] == blockid) {
								overlap.push_back(r.from());
                            }
                            else {
                                mark_dirty(r.from());
                            }
                        }
                    }
                    for (auto s : overlap) {
						mark_dirty(s);
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

