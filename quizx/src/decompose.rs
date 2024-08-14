// QuiZX - Rust library for quantum circuit rewriting and optimisation
//         using the ZX-calculus
// Copyright (C) 2021 - Aleks Kissinger
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::graph::*;
use crate::scalar::*;
// use hashbrown::HashMap;
// use hashbrown::HashSet;
use hopcroft_karp::matching;
use itertools::Itertools;
use num::Rational64;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rayon::vec;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SimpFunc {
    FullSimp,
    CliffordSimp,
    NoSimp,
}
use SimpFunc::*;

/// Store the (partial) decomposition of a graph into stabilisers
#[derive(Clone)]
pub struct Decomposer<G: GraphLike> {
    pub stack: VecDeque<(usize, G)>,
    pub done: Vec<G>,
    pub scalar: ScalarN,
    pub nterms: usize,
    simp_func: SimpFunc,
    random_t: bool,
    use_cats: bool,
    use_heur: bool,
    use_paired_heur: bool,
    use_sub_comp: bool,
    split_comps: bool,
    is_bench: bool,
    save: bool, // save graphs on 'done' stack
    max_depth: usize,
}

// impl<G: GraphLike> Send for Decomposer<G> {}

/// Gives upper bound for number of terms needed for BSS decomposition
///
/// Note this number can be very large. We use a float here to avoid overflows.
pub fn terms_for_tcount(tcount: usize) -> f64 {
    let mut t = tcount as i32;
    let mut count = 7f64.powi(t / 6i32);
    t %= 6;
    count *= 2f64.powi(t / 2i32);
    if t % 2 == 1 {
        count *= 2.0;
    }
    count
}

impl<G: GraphLike> Decomposer<G> {
    pub fn empty() -> Decomposer<G> {
        Decomposer {
            stack: VecDeque::new(),
            done: vec![],
            scalar: ScalarN::zero(),
            nterms: 0,
            simp_func: NoSimp,
            random_t: false,
            use_cats: false,
            use_heur: false,
            use_paired_heur: false,
            use_sub_comp: false,
            split_comps: false,
            is_bench: false,
            save: false,
            max_depth: 0,
        }
    }

    pub fn new(g: &G) -> Decomposer<G> {
        let mut d = Decomposer::empty();
        d.stack.push_back((0, g.clone()));
        d
    }

    /// Split a Decomposer with N graphs on the stack into N Decomposers
    /// with 1 graph each.
    ///
    /// Used for parallelising. The last decomposer in the list keeps the
    /// current state (e.g. `nterms` and `scalar`).
    pub fn split(mut self) -> Vec<Decomposer<G>> {
        let mut ds = vec![];
        while self.stack.len() > 1 {
            let (_, g) = self.stack.pop_front().unwrap();
            let mut d1 = Decomposer::new(&g);
            d1.save(self.save)
                .random_t(self.random_t)
                .with_simp(self.simp_func)
                .split_comps(self.split_comps)
                .use_cats(self.use_cats)
                .use_heur(self.use_heur)
                .use_paired_heur(self.use_paired_heur);
            ds.push(d1);
        }
        ds.push(self);
        ds
    }

    /// Merge N decomposers into 1, adding scalars together
    pub fn merge(mut ds: Vec<Decomposer<G>>) -> Decomposer<G> {
        if let Some(mut d) = ds.pop() {
            while let Some(d1) = ds.pop() {
                d.scalar = d.scalar + d1.scalar;
                d.nterms += d1.nterms;
                d.stack.extend(d1.stack);
                d.done.extend(d1.done);
            }
            d
        } else {
            Decomposer::empty()
        }
    }

    // pub fn seed(&mut self, seed: u64) -> &mut Self { self.rng = StdRng::seed_from_u64(seed); self }

    pub fn with_simp(&mut self, f: SimpFunc) -> &mut Self {
        self.simp_func = f;
        self
    }

    pub fn with_full_simp(&mut self) -> &mut Self {
        self.with_simp(FullSimp)
    }
    pub fn with_clifford_simp(&mut self) -> &mut Self {
        self.with_simp(CliffordSimp)
    }

    pub fn random_t(&mut self, b: bool) -> &mut Self {
        self.random_t = b;
        self
    }

    pub fn use_cats(&mut self, b: bool) -> &mut Self {
        self.use_cats = b;
        self
    }

    pub fn use_heur(&mut self, b: bool) -> &mut Self {
        self.use_heur = b;
        self.use_paired_heur = b;
        self
    }

    pub fn use_paired_heur(&mut self, b: bool) -> &mut Self {
        self.use_heur |= b;
        self.use_paired_heur = b;
        self
    }

    pub fn use_sub_comp(&mut self, b: bool) -> &mut Self {
        self.use_sub_comp = b;
        self
    }

    pub fn split_comps(&mut self, b: bool) -> &mut Self {
        self.split_comps = b;
        self
    }

    pub fn is_bench(&mut self, b: bool) -> &mut Self {
        self.is_bench = b;
        self
    }

    pub fn save(&mut self, b: bool) -> &mut Self {
        self.save = b;
        self
    }

    pub fn max_depth(&mut self, d: usize) -> &mut Self {
        self.max_depth = d;
        self
    }

    /// Computes `terms_for_tcount` for every graph on the stack
    pub fn max_terms(&self) -> f64 {
        let mut n = 0.0;
        for (_, g) in &self.stack {
            n += terms_for_tcount(g.tcount());
        }

        n
    }

    pub fn pop_graph(&mut self) -> G {
        let (_, g) = self.stack.pop_back().unwrap();
        g
    }

    /// Decompose the first <= 6 T gates in the graph on the top of the
    /// stack.
    pub fn decomp_top(&mut self) -> &mut Self {
        let (depth, g) = self.stack.pop_back().unwrap();
        if self.split_comps {
            let g_comps = Decomposer::split_components(&g);
            if g_comps.len() > 1 {
                return self.decomp_split_comps(&g, &g_comps, depth);
            }
        }

        if self.use_sub_comp {
            let vs = Decomposer::sub_comp(&g).into_iter().collect_vec();
            if !vs.is_empty() {
                return self.push_decomp(
                    &[
                        Decomposer::replace_sub_comp_0,
                        Decomposer::replace_sub_comp_1,
                    ],
                    depth + 1,
                    &g,
                    &vs,
                );
            }
        }

        // for e in g.edges() {
        //     println!("{:?}", e.2);
        // }

        if self.is_bench && depth % 5 == 0 {
            return self.decomp_bench_cut(depth, &g);
        }

        if self.use_cats {
            let cat_nodes = Decomposer::cat_ts(&g); //gadget_ts(&g);
                                                    //println!("{:?}", gadget_nodes);
                                                    //let nts = cat_nodes.iter().fold(0, |acc, &x| if g.phase(x).denom() == &4 { acc + 1 } else { acc });
                                                    // if !cat_nodes.is_empty() {
                                                    //     // println!("using cat!");
                                                    //     return self.push_cat_decomp(depth + 1, &g, &cat_nodes);
                                                    // }
            let nts = cat_nodes.iter().fold(0, |acc, &x| {
                if g.phase(x).denom() == &4 {
                    acc + 1
                } else {
                    acc
                }
            });

            // if nts == 4 {
            //     // println!("using cat!");
            //     return self.push_cat_decomp(depth + 1, &g, &cat_nodes);
            // }

            // println!("new graph");

            if self.use_heur {
                let (vs, eff_a) = Decomposer::cut_v(&g);
                // let vs_pg = Decomposer::cut_pg(&g);
                let vs_pg = vec![];
                let (vs_pair, eff_a_pair) = if self.use_paired_heur {
                    Decomposer::cut_v_pair(&g)
                } else {
                    (vec![], -1.0)
                };
                // let (vs_pair, eff_a_pair) = Decomposer::cut_v_pair(&g);
                // let (vs_pair, eff_a_pair) = (vec![], -1.0);

                let mut should_cut_v = eff_a > 0.0;
                let mut should_cut_v_pair = eff_a_pair > 0.0;
                let should_cut_pg = !vs_pg.is_empty();
                let mut bet_eff_a = f64::MAX;
                let mut bet_vs = vec![];

                if should_cut_v && eff_a < bet_eff_a {
                    bet_eff_a = eff_a;
                    bet_vs = vs;
                }
                if should_cut_v_pair && eff_a_pair < bet_eff_a {
                    bet_eff_a = eff_a_pair;
                    bet_vs = vs_pair;
                    should_cut_v = false;
                }
                if should_cut_pg && 0.25 < bet_eff_a {
                    bet_eff_a = 0.25;
                    bet_vs = vs_pg;
                    should_cut_v = false;
                    should_cut_v_pair = false;
                }

                if bet_eff_a > 0.0
                    && ((nts == 4 && bet_eff_a < 0.25)
                        || (nts == 6 && bet_eff_a < 0.264)
                        || (nts == 5 && bet_eff_a < 0.316)
                        || (nts == 3 && bet_eff_a < 0.333)
                        || ((nts > 6 || nts < 3) && bet_eff_a < 0.4))
                {
                    if should_cut_v {
                        // println!("cutting v, eff_a: {:?}", bet_eff_a);
                        return self.push_decomp(
                            &[Decomposer::replace_t0, Decomposer::replace_t1],
                            depth + 1,
                            &g,
                            &bet_vs,
                        );
                    } else if should_cut_v_pair {
                        // println!("cutting cat4, eff_a: {:?}", bet_eff_a);
                        return self.push_decomp(
                            &[Decomposer::replace_tpair0, Decomposer::replace_tpair1],
                            depth + 1,
                            &g,
                            &bet_vs,
                        );
                    } else if should_cut_pg {
                        // println!("cutting pg, eff_a: {:?}", bet_eff_a);
                        return self.push_decomp(
                            &[Decomposer::cut_pg_0, Decomposer::cut_pg_0],
                            depth + 1,
                            &g,
                            &bet_vs,
                        );
                    }
                }
            }

            if !cat_nodes.is_empty() {
                // println!("using cat!");
                return self.push_cat_decomp(depth + 1, &g, &cat_nodes);
            }

            let ts = Decomposer::first_ts(&g);
            if ts.len() >= 5 {
                return self.push_magic5_from_cat_decomp(depth + 1, &g, &ts[..5]);
            }
        }
        let ts = if self.random_t {
            Decomposer::random_ts(&g, &mut thread_rng())
        } else {
            Decomposer::first_ts(&g)
        };
        self.decomp_ts(depth, g, &ts);
        self
    }

    /// Decompose until there are no T gates left
    pub fn decomp_all(&mut self) -> &mut Self {
        while !self.stack.is_empty() {
            self.decomp_top();
        }
        self
    }

    /// Decompose breadth-first until the given depth
    pub fn decomp_until_depth(&mut self, depth: usize) -> &mut Self {
        while !self.stack.is_empty() {
            // pop from the bottom of the stack to work breadth-first
            let (d, g) = self.stack.pop_front().unwrap();
            if d >= depth {
                self.stack.push_front((d, g));
                break;
            } else {
                if self.split_comps {
                    let g_comps = Decomposer::split_components(&g);
                    if g_comps.len() > 1 {
                        return self.decomp_split_comps(&g, &g_comps, d);
                    }
                }
                if self.use_sub_comp {
                    let vs = Decomposer::sub_comp(&g).into_iter().collect_vec();
                    if !vs.is_empty() {
                        return self.push_decomp(
                            &[
                                Decomposer::replace_sub_comp_0,
                                Decomposer::replace_sub_comp_1,
                            ],
                            depth + 1,
                            &g,
                            &vs,
                        );
                    }
                }

                if self.is_bench && d % 5 == 0 {
                    return self.decomp_bench_cut(d, &g);
                }

                if self.use_cats {
                    let cat_nodes = Decomposer::cat_ts(&g); //gadget_ts(&g);
                                                            //println!("{:?}", gadget_nodes);
                    let nts = cat_nodes.iter().fold(0, |acc, &x| {
                        if g.phase(x).denom() == &4 {
                            acc + 1
                        } else {
                            acc
                        }
                    });

                    if self.use_heur {
                        let (vs, eff_a) = Decomposer::cut_v(&g);
                        // let vs_pg = Decomposer::cut_pg(&g);
                        let vs_pg = vec![];
                        let (vs_pair, eff_a_pair) = if self.use_paired_heur {
                            Decomposer::cut_v_pair(&g)
                        } else {
                            (vec![], -1.0)
                        };
                        // let (vs_pair, eff_a_pair) = (vec![], -1.0);

                        let mut should_cut_v = eff_a > 0.0;
                        let mut should_cut_v_pair = eff_a_pair > 0.0;
                        let should_cut_pg = !vs_pg.is_empty();
                        let mut bet_eff_a = f64::MAX;
                        let mut bet_vs = vec![];

                        if should_cut_v && eff_a < bet_eff_a {
                            bet_eff_a = eff_a;
                            bet_vs = vs;
                        }
                        if should_cut_v_pair && eff_a_pair < bet_eff_a {
                            bet_eff_a = eff_a_pair;
                            bet_vs = vs_pair;
                            should_cut_v = false;
                        }
                        if should_cut_pg && 0.25 < bet_eff_a {
                            bet_eff_a = 0.25;
                            bet_vs = vs_pg;
                            should_cut_v = false;
                            should_cut_v_pair = false;
                        }

                        if bet_eff_a > 0.0
                            && ((nts == 4 && bet_eff_a < 0.25)
                                || (nts == 6 && bet_eff_a < 0.264)
                                || (nts == 5 && bet_eff_a < 0.316)
                                || (nts == 3 && bet_eff_a < 0.333)
                                || ((nts > 6 || nts < 3) && bet_eff_a < 0.4))
                        {
                            if should_cut_v {
                                // println!("cutting v, eff_a: {:?}", bet_eff_a);
                                return self.push_decomp(
                                    &[Decomposer::replace_t0, Decomposer::replace_t1],
                                    depth + 1,
                                    &g,
                                    &bet_vs,
                                );
                            } else if should_cut_v_pair {
                                // println!("cutting cat4, eff_a: {:?}", bet_eff_a);
                                return self.push_decomp(
                                    &[Decomposer::replace_tpair0, Decomposer::replace_tpair1],
                                    depth + 1,
                                    &g,
                                    &bet_vs,
                                );
                            } else if should_cut_pg {
                                // println!("cutting pg, eff_a: {:?}", bet_eff_a);
                                return self.push_decomp(
                                    &[Decomposer::cut_pg_0, Decomposer::cut_pg_0],
                                    depth + 1,
                                    &g,
                                    &bet_vs,
                                );
                            }
                        }
                    }

                    if !cat_nodes.is_empty() {
                        // println!("using cat!");
                        return self.push_cat_decomp(depth + 1, &g, &cat_nodes);
                    }

                    let ts = Decomposer::first_ts(&g);
                    if ts.len() >= 5 {
                        return self.push_magic5_from_cat_decomp(depth + 1, &g, &ts[..5]);
                    }
                }
                let ts = if self.random_t {
                    Decomposer::random_ts(&g, &mut thread_rng())
                } else {
                    Decomposer::first_ts(&g)
                };
                self.decomp_ts(d, g, &ts);
            }
        }
        self
    }

    /// Decompose in parallel, starting at the given depth
    pub fn decomp_parallel(mut self, depth: usize) -> Self {
        self.max_depth(depth);
        self.decomp_until_depth(depth);
        let ds = self.split();
        Decomposer::merge(
            ds.into_par_iter()
                .map(|mut d| {
                    d.decomp_all();
                    d
                })
                .collect(),
        )
    }

    pub fn decomp_ts(&mut self, depth: usize, g: G, ts: &[usize]) {
        if ts.len() == 6 {
            self.push_bss_decomp(depth + 1, &g, ts);
        } else if ts.len() >= 2 {
            self.push_sym_decomp(depth + 1, &g, &ts[0..2]);
        } else if !ts.is_empty() {
            self.push_single_decomp(depth + 1, &g, ts);
        } else {
            // crate::simplify::full_simp(&mut g);
            self.scalar = &self.scalar + g.scalar();
            self.nterms += 1;
            if g.num_vertices() != 0 {
                println!("{}", g.to_dot());
                println!("WARNING: graph was not fully reduced");
                // println!("{}", g.to_dot());
            }
            if self.save {
                self.done.push(g);
            }
        }
    }

    pub fn decomp_split_comps(&mut self, g: &G, g_comps: &Vec<G>, depth: usize) -> &mut Self {
        let mut nterms_comps = 0;
        let mut scalar_comps = g.scalar().clone();
        for h in g_comps {
            let mut d = Decomposer::new(h);
            d.use_cats(self.use_cats);
            d.split_comps(self.split_comps);
            d.use_heur(self.use_heur);
            d.use_paired_heur(self.use_paired_heur);
            d.with_full_simp();

            let depth = if depth > self.max_depth {
                depth - self.max_depth
            } else {
                0
            };

            let d = d.decomp_parallel(depth);

            nterms_comps += d.nterms;
            scalar_comps *= d.scalar;
        }
        self.nterms += nterms_comps;
        self.scalar = &self.scalar + scalar_comps;

        self
    }

    pub fn decomp_bench_cut(&mut self, depth: usize, g: &G) -> &mut Self {
        let mut vertices_on_legs: Vec<V> = g
            .vertices()
            .filter(|&v| g.phase(v).denom() == &4 && g.neighbor_vec(v).len() > 1)
            .collect_vec();
        vertices_on_legs.sort_unstable_by(|a, b| {
            let mut cnt_a = 0usize;
            let mut cnt_b = 0usize;
            for v in g.neighbor_vec(*a) {
                let v_neigh = g.neighbor_vec(v);
                if g.phase(v).denom() == &1 && v_neigh.len() == 3 {
                    cnt_a += 2;
                } else if g.phase(v).denom() == &4 && v_neigh.len() == 1 {
                    cnt_a += 1;
                } else if g.phase(v).denom() == &4 && v_neigh.len() == 2 {
                    for w in v_neigh {
                        if w == *a {
                            continue;
                        }
                        if g.phase(w).denom() == &4 && g.neighbor_vec(w).len() == 1 {
                            cnt_a += 2;
                        }
                    }
                }
            }

            for v in g.neighbor_vec(*b) {
                let v_neigh = g.neighbor_vec(v);
                if g.phase(v).denom() == &1 && v_neigh.len() == 3 {
                    cnt_b += 2;
                } else if g.phase(v).denom() == &4 && v_neigh.len() == 1 {
                    cnt_b += 1;
                } else if g.phase(v).denom() == &4 && v_neigh.len() == 2 {
                    for w in v_neigh {
                        if w == *a {
                            continue;
                        }
                        if g.phase(w).denom() == &4 && g.neighbor_vec(w).len() == 1 {
                            cnt_b += 2;
                        }
                    }
                }
            }
            match cnt_b.cmp(&cnt_a) {
                std::cmp::Ordering::Equal => {
                    let n_neigh_a = g.neighbor_vec(*a).len();
                    let n_neigh_b = g.neighbor_vec(*b).len();
                    n_neigh_b.cmp(&n_neigh_a)
                }
                x => x,
            }
        });
        let vertices_on_legs = vertices_on_legs.into_iter().take(5).collect_vec();
        let mut v_metrics = HashMap::new();
        let mut best_n_terms = usize::MAX;
        // let mut best_decomp = None;
        let mut best_nv = vec![];
        for v in vertices_on_legs.clone() {
            let mut cv_metrics = HashMap::new();
            cv_metrics.insert(0, 0usize);
            cv_metrics.insert(1, 0usize);
            cv_metrics.insert(2, 0usize);
            for w in g.neighbor_vec(v) {
                if g.phase(w).denom() == &4 {
                    if g.neighbor_vec(w).len() == 1 {
                        cv_metrics.entry(1).and_modify(|e| *e += 1);
                    } else if g.neighbor_vec(w).len() == 2 {
                        for x in g.neighbor_vec(w) {
                            if x == v {
                                continue;
                            }
                            if g.phase(x).denom() == &4 && g.neighbor_vec(x).len() == 1 {
                                cv_metrics.entry(2).and_modify(|e| *e += 2);
                            }
                        }
                    }
                    cv_metrics.entry(0).and_modify(|e| *e += 1);
                } else {
                    let n = g.neighbor_vec(w).len();
                    if !cv_metrics.contains_key(&n) {
                        cv_metrics.insert(n, 0usize);
                    }
                    cv_metrics.entry(n).and_modify(|e| *e += 1);
                }
            }
            v_metrics.insert(v, cv_metrics);

            let nv = vec![v];
            // nv.extend(g.neighbor_vec(v));

            let mut h = Decomposer::new(g);
            h.use_cats(self.use_cats);
            h.split_comps(self.split_comps);
            h.use_heur(self.use_heur);
            h.is_bench(self.is_bench);
            h.with_full_simp();

            h.pop_graph();
            h.push_decomp(
                &[Decomposer::replace_t0, Decomposer::replace_t1],
                depth + 1,
                &g,
                &nv,
            );

            let h = h.decomp_parallel(3);

            if h.nterms < best_n_terms {
                best_n_terms = h.nterms;
                // best_decomp = Some(h);
                best_nv = nv;
            }
        }

        if best_n_terms < usize::MAX {
            if depth == 0 {
                // println!("{:?}", best_nv[0]);
                println!("");
                for v in vertices_on_legs {
                    if v == best_nv[0] {
                        println!("best: {:?}", v);
                    }
                    println!("{:?}", v_metrics.get(&v).unwrap());
                }
            }
            return self.push_decomp(
                &[Decomposer::replace_t0, Decomposer::replace_t1],
                depth + 1,
                &g,
                &best_nv,
            );
        }
        self
    }

    /// Pick the first <= 6 T gates from the given graph
    pub fn first_ts(g: &G) -> Vec<V> {
        let mut t = vec![];

        for v in g.vertices() {
            if *g.phase(v).denom() == 4 {
                t.push(v);
            }
            if t.len() == 6 {
                break;
            }
        }

        t
    }

    /// Pick <= 6 T gates from the given graph, chosen at random
    pub fn random_ts(g: &G, rng: &mut impl Rng) -> Vec<V> {
        let mut all_t: Vec<_> = g.vertices().filter(|&v| *g.phase(v).denom() == 4).collect();
        let mut t = vec![];

        while t.len() < 6 && !all_t.is_empty() {
            let i = rng.gen_range(0..all_t.len());
            t.push(all_t.swap_remove(i));
        }

        t
    }

    // Subgraph complement
    fn sub_comp(g: &G) -> FxHashSet<V> {
        // Get T spiders not in phase gadgets
        let in_vs: FxHashSet<V> = g
            .vertices()
            .filter(|&v| g.phase(v).denom() == &4 && g.neighbor_vec(v).len() > 1)
            .collect();
        let n = in_vs.len();
        if n < 2 {
            return FxHashSet::default();
        }

        // > 3/4 complete, 3*n(n-1)/8 edges
        let n_thresh = (3 * n * (n - 1)) >> 3;
        let sg = g.induced_subgraph(&in_vs);
        let n_edges = sg.num_edges();
        let vs = if n_edges > n_thresh {
            in_vs
        } else {
            FxHashSet::default()
        };
        vs
    }

    // fn jaccard_similarity(set1: &[V], set2: &[V]) -> f64 {
    //     let set1: HashSet<_> = set1.iter().cloned().collect();
    //     let set2: HashSet<_> = set2.iter().cloned().collect();

    //     let intersection: HashSet<_> = set1.intersection(&set2).collect();
    //     let union: HashSet<_> = set1.union(&set2).collect();

    //     intersection.len() as f64 / union.len() as f64
    // }

    // Cut paired v
    pub fn cut_v_pair(g: &G) -> (Vec<V>, f64) {
        // for v in g.vertices() {
        //     let v_neigh = g.neighbor_vec(v);
        //     if g.phase(v).denom() != &1 || v_neigh.len() != 4 {
        //         continue;
        //     }

        //     let v_neigh_set = v_neigh.iter().cloned().collect::<HashSet<_>>();

        //     for w in g.vertices() {
        //         let w_neigh = g.neighbor_vec(w);
        //         if g.phase(w).denom() != &1 || w_neigh.len() != 4 {
        //             continue;
        //         }

        //         let w_neigh_set = w_neigh.iter().cloned().collect::<HashSet<_>>();

        //         let mut common = v_neigh_set.intersection(&w_neigh_set).cloned().collect_vec();

        //         if common.len() != 2 {
        //             continue;
        //         }

        //         let mut vs = vec![v, w];
        //         vs.append(&mut common);
        //         return vs;

        //     }
        // }

        let mut best_n = 0usize;
        let mut best_vs = vec![];
        let mut eff_a = -1.0;
        for v in g.vertices() {
            if g.phase(v).denom() != &4 {
                continue;
            }
            let v_neigh0 = g
                .neighbor_vec(v)
                .iter()
                .cloned()
                .filter(|&w| g.phase(w).denom() == &1 && g.neighbor_vec(w).len() == 4)
                .collect_vec();
            let v_neight = g
                .neighbor_vec(v)
                .iter()
                .cloned()
                .filter(|&w| g.phase(w).denom() == &4 && g.neighbor_vec(w).len() == 2)
                .collect_vec();


            let v_neigh0_set = v_neigh0.iter().cloned().collect::<HashSet<_>>();
            let v_neight_set = v_neight.iter().cloned().collect::<HashSet<_>>();

            for w in g.vertices() {
                if g.phase(w).denom() != &4 || w == v {
                    continue;
                }
                let w_neigh0 = g
                    .neighbor_vec(w)
                    .iter()
                    .cloned()
                    .filter(|&w| g.phase(w).denom() == &1 && g.neighbor_vec(w).len() == 4)
                    .collect_vec();
                let w_neight = g
                    .neighbor_vec(w)
                    .iter()
                    .cloned()
                    .filter(|&w| g.phase(w).denom() == &4 && g.neighbor_vec(w).len() == 2)
                    .collect_vec();

                let w_neigh0_set = w_neigh0.iter().cloned().collect::<HashSet<_>>();
                let w_neight_set = w_neight.iter().cloned().collect::<HashSet<_>>();

                let common0 = v_neigh0_set
                    .intersection(&w_neigh0_set)
                    .cloned()
                    .collect_vec();
                let commont = v_neight_set
                    .intersection(&w_neight_set)
                    .cloned()
                    .collect_vec();

                let n0 = 2 * common0.len();
                let nt = commont.len();

                if n0 <= best_n && nt <= best_n {
                    continue;
                }

                if n0 >= nt {
                    best_n = n0;
                    best_vs = common0;
                    best_vs.append(vec![v, w].as_mut());
                    eff_a = 1.0 / (n0 + 2) as f64;
                } else {
                    best_n = nt;
                    best_vs = commont;
                    best_vs.append(vec![v, w].as_mut());
                    eff_a = 1.0 / (nt + 2) as f64;
                }
            }
        }

        (best_vs, eff_a)
    }

    fn uncommon_elements(a: &HashSet<V>, b: &HashSet<V>) -> HashSet<V> {
        a.union(b)
            .cloned()
            .collect::<HashSet<_>>()
            .difference(&a.intersection(b).cloned().collect::<HashSet<_>>())
            .cloned()
            .collect()
    }

    pub fn cut_pg(g: &G) -> Vec<V> {
        let mut processed_pairs = HashSet::new();

        // Add weight for T-spiders in a pair
        for v in g.vertices() {
            let v_neigh = g.neighbor_vec(v);
            // let n = v_neigh.len();
            if *g.phase(v).denom() != 1 {
                continue;
            }
            for w in g.vertices() {
                if g.phase(w).denom() != &1 {
                    continue;
                }
                let pair = (v.min(w), v.max(w));
                if v == w || processed_pairs.contains(&pair) {
                    continue;
                }
                processed_pairs.insert(pair);

                let w_neigh_set = g.neighbor_vec(w).iter().cloned().collect::<HashSet<_>>();
                let v_neigh_set = v_neigh.iter().cloned().collect::<HashSet<_>>();

                // phase gadget heuristic
                let uncommon = Self::uncommon_elements(&v_neigh_set, &w_neigh_set)
                    .into_iter()
                    .collect::<Vec<_>>();
                if uncommon.len() != 4 {
                    continue;
                }

                // let (x0, x1, x2) = uncommon.iter().collect_tuple().unwrap();
                // let (n0, n1, n2) = uncommon.iter().map(|x| g.neighbor_vec(*x).len()).collect_tuple().unwrap();

                // let bv = uncommon.iter().map(|x| v_neigh_set.contains(x)).collect_vec();
                // let bw = uncommon.iter().map(|x| w_neigh_set.contains(x)).collect_vec();
                let bv = uncommon
                    .iter()
                    .filter(|x| v_neigh_set.contains(x))
                    .cloned()
                    .collect_vec();
                let bw = uncommon
                    .iter()
                    .filter(|x| w_neigh_set.contains(x))
                    .cloned()
                    .collect_vec();

                let mut poss_cut: Vec<V>;
                let cv: V;

                if bv.len() > 1 {
                    cv = v;
                    poss_cut = bv;
                } else {
                    cv = w;
                    poss_cut = bw;
                };

                poss_cut.sort_by_key(|v| g.neighbor_vec(*v).len());

                poss_cut.insert(0, cv);

                return poss_cut;
                
            }
        }
        return vec![];
    }

    pub fn cut_v(g: &G) -> (Vec<V>, f64) {
        let mut vertices_with_denom_1 = HashMap::new();
        let mut vertices_with_denom_4 = HashMap::new();
        let mut weights = HashMap::new();
        let mut weights5 = HashMap::new();
        // let mut weights2 = HashMap::new();

        for v in g.vertices() {
            if *g.phase(v).denom() == 1 {
                let neighbours = g.neighbor_vec(v);
                let filtered_neighbours = neighbours
                    .iter()
                    .filter(|&w| g.neighbor_vec(*w).len() > 1)
                    .cloned()
                    .collect::<HashSet<_>>();
                vertices_with_denom_1.insert(v, filtered_neighbours);
            } else if *g.phase(v).denom() == 4 {
                // ignore the ones in the gadgets
                if g.neighbor_vec(v).len() < 2 {
                    continue;
                }
                let filtered_neighbours = g
                    .neighbor_vec(v)
                    .into_iter()
                    .filter(|&w| *g.phase(w).denom() == 1)
                    .collect::<HashSet<_>>();
                // if g.neighbor_vec(v).len() >= 2 {
                vertices_with_denom_4.insert(v, filtered_neighbours);
                weights.insert(v, 0.0);
                weights5.insert(v, 0.0);
                // weights2.insert(v, 0.0);
                // }
            }
        }
        // let mut processed_pairs = HashSet::new();

        // Add weight for T-spiders in a pair
        for v in g.vertices() {
            let v_neigh = g.neighbor_vec(v);
            let n = v_neigh.len();
            if *g.phase(v).denom() == 1 {
                if n == 3 {
                    // cat3 heuristic
                    for w in v_neigh.clone() {
                        weights.entry(w).and_modify(|e| *e += 2.0);
                    }
                    // cat3 vertices should not be part of phase gadget cuts
                    continue;
                } else if n == 5 {
                    // cat5 heuristic
                    for w in v_neigh.clone() {
                        weights5.entry(w).and_modify(|e| *e += 1.0);
                    }
                }

                // for w in g.vertices() {
                //     if g.phase(w).denom() != &1 || g.neighbor_vec(w).len() == 3 {
                //         continue;
                //     }
                //     let pair = (v.min(w), v.max(w));
                //     if v == w || processed_pairs.contains(&pair) {
                //         continue;
                //     }
                //     processed_pairs.insert(pair);

                //     let w_neigh_set = g.neighbor_vec(w).iter().cloned().collect::<HashSet<_>>();
                //     let v_neigh_set = v_neigh.iter().cloned().collect::<HashSet<_>>();

                //     // phase gadget heuristic
                //     let uncommon = Self::uncommon_elements(&v_neigh_set, &w_neigh_set).into_iter().collect::<Vec<_>>();
                //     if uncommon.len() != 4 {
                //         continue;
                //     }

                //     // let (x0, x1, x2) = uncommon.iter().collect_tuple().unwrap();
                //     // let (n0, n1, n2) = uncommon.iter().map(|x| g.neighbor_vec(*x).len()).collect_tuple().unwrap();

                //     // let bv = uncommon.iter().map(|x| v_neigh_set.contains(x)).collect_vec();
                //     // let bw = uncommon.iter().map(|x| w_neigh_set.contains(x)).collect_vec();
                //     let bv = uncommon.iter().map(|x| v_neigh_set.contains(x)).filter(|x| *x);
                //     let bw = uncommon.iter().map(|x| w_neigh_set.contains(x)).filter(|x| *x);

                //     // let mut poss_cut = vec![];

                //     let mut poss_cut = if bv.count() > 1 {
                //         bv.collect_vec()
                //     } else {
                //         bw.collect_vec()
                //     };

                //     poss_cut.sort_by_key(|v| g.neighbor_vec(*v).len());

                // for i in 0..3 {
                //     for j in i..3 {
                //         if i == j {
                //             continue;
                //         }
                //         if (bv[i] && bv[j]) || (bw[i] && bw[j]) {
                //             poss_cut = vec![uncommon[i], uncommon[j]];
                //             break;
                //         }
                //     }
                // }

                // let poss_cut_n = poss_cut.iter().map(|x| g.neighbor_vec(*x).len() == 1).collect_vec();
                // if poss_cut_n[0] && poss_cut_n[1] {
                //     weights.entry(poss_cut[0]).and_modify(|e| *e += 2.0);
                //     weights.entry(poss_cut[1]).and_modify(|e| *e += 2.0);
                // } else if poss_cut_n[0] {
                //     weights.entry(poss_cut[1]).and_modify(|e| *e += 2.0);
                // } else if poss_cut_n[1] {
                //     weights.entry(poss_cut[0]).and_modify(|e| *e += 2.0);
                // }
                // }

                // if n != 3 {
                //     continue;
                // }

                // for w in v_neigh {
                //     weights.entry(w).and_modify(|e| *e += 2.0 / ((n - 2) as f64));
                // }
                // weights.entry(v).and_modify(|e| *e += 1.0);
            } else {
                // Removing itself from the cut
                weights.entry(v).and_modify(|e| *e += 1.0);
                if n > 1 {
                    continue;
                }
                // Lone phase heuristic
                for w in v_neigh {
                    weights.entry(w).and_modify(|e| *e += 1.0);
                }
            }
        }

        // let mut union_nn = HashMap::new();
        // let mut ncut_n = HashMap::new();
        // let mut num_n_all = HashMap::new();
        // let mut bi_match = HashMap::new();

        // let max_catn = 3;

        // for (v, v_neigh) in vertices_with_denom_4.clone() {
        //     let mut uv: HashSet<V> = HashSet::new();
        //     // let mut uv: HashMap<usize, HashSet<V>> = HashMap::new();
        //     // let mut num_n = HashMap::new();
        //     // ncut_n.insert(v, 0.0);
        //     // let mut v_edges = vec![];
        //     for w in v_neigh {
        //         // let w_neigh = vertices_with_denom_1.get(&w).unwrap();
        //         let w_neigh = g.neighbor_vec(w);
        //         let n = w_neigh.len();
        //         // println!("n: {}", n);
        //         // limit how long the cut sequence is
        //         if n > max_catn {
        //             continue;
        //         }

        //         // for x in w_neigh.clone() {
        //         //     if g.neighbor_vec(x).len() < 2 {
        //         //         continue;
        //         //     }
        //         //     v_edges.push((w, x));
        //         // }

        //         // if !uv.contains_key(&n) {
        //         //     uv.insert(n, HashSet::new());
        //         //     num_n.insert(n, 0);
        //         // }

        //         // uv.entry(n).and_modify(|f| f.extend(w_neigh));
        //         // num_n.entry(n).and_modify(|f| *f += 1);

        //         uv.extend(w_neigh);
        //         // ncut_n.entry(v).and_modify(|e| *e += (n - 3) as f64);

        //         // To signify the reduction of terms in catn decomposition
        //         // if n == 5 {
        //         //     println!("n == 5");
        //         //     ncut_n.entry(v).and_modify(|e| *e += 1.0 - (3.0_f64).log2());
        //         // } else if n == 7 {
        //         //     println!("n == 7");
        //         //     ncut_n.entry(v).and_modify(|e| *e -= 1.0);
        //         // }
        //     }
        //     union_nn.insert(v, uv);
        //     // num_n_all.insert(v, num_n);
        //     // ncut_n.entry(v).and_modify(|e| *e += 1.0);

        //     // let res_match = matching(&v_edges);
        //     // bi_match.insert(v, res_match.len());
        // }

        // // let mut tts = HashMap::new();

        // for (v, uv) in union_nn {
        //     // num cuts needed
        //     // let ncut = *ncut_n.get(&v).unwrap();
        //     // T reduced
        //     let k = vertices_with_denom_4.get(&v).unwrap().into_iter().filter(
        //         |&w| g.neighbor_vec(*w).len() <= max_catn
        //     ).count();
        //     if k < 2 {
        //         continue;
        //     }
        //     // // // let rt = uv.len() + num_seq_n_or_less;
        //     let rt = uv.len();
        //     // higher degree vertices have more weight
        //     // let frt = rt as f64 + 0.1 * g.neighbor_vec(v).len() as f64;
        //     // let ncuts = (rt - k - bi_match.get(&v).unwrap()) as f64;
        //     let ncuts = (rt - 2*k) as f64;
        //     // tts.insert(v, rt);
        //     // println!("rt: {}, k: {}", rt, k);

        //     // let mut best_rt = 0usize;
        //     // let mut best_k = usize::MAX;
        //     // let mut best_weight = 0.0f64;
        //     // let mut cur_uv: HashSet<V> = HashSet::new();
        //     // let mut k = 0usize;
        //     // let num_n = num_n_all.get(&v).unwrap();

        //     // let mut keys = uv.keys().collect::<Vec<_>>();
        //     // keys.sort_unstable();

        //     // for n in keys {
        //     //     let u = uv.get(n).unwrap();
        //     //     // let rt = u.len();
        //     //     let n_k = num_n.get(n).unwrap();

        //     //     let rt_before = cur_uv.len();
        //     //     cur_uv.extend(u);
        //     //     let rt_after = cur_uv.len();

        //     //     k += n_k;

        //     //     let rt = cur_uv.len();

        //     //     let nlegs = (*n - 2) as f64;

        //     //     let mul = 1.0 + 1.0 / nlegs;
        //     //     let k_mul = k as f64 * mul;

        //     //     let weight = rt as f64 / (rt as f64 - k_mul);
        //     //     best_weight = best_weight.max(weight);
        //     //     // if weight > best_weight {
        //     //     //     best_weight = weight;
        //     //     //     best_rt = rt;
        //     //     //     best_k = k;
        //     //     // }
        //     // }

        //     // weights.insert(v, best_weight);
        //     weights.insert(v, rt as f64 / ncuts);

        //     // let h1 = weights.get(&v).unwrap();
        //     // let h2 = weights2.get(&v).unwrap();
        //     // if *h1 >= 1.0 || *h2 > 1.0 {
        //     //     println!("v: {}, v2: {}", h1, h2);
        //     // }
        // }

        // Initialize a hashmap to track uncommon elements between groups
        // Keys: Pairs of 0-spiders, Values: Uncommon T-spiders
        // let mut group_uncommon: HashMap<(V, V), HashSet<V>> = HashMap::new();
        // for v in g.vertices() {
        //     if *g.phase(v).denom() != 1 {
        //         continue;
        //     }
        //     let v_neigh : HashSet<V> = g.neighbor_vec(v).iter().cloned().collect();
        //     for w in g.vertices() {
        //         if *g.phase(w).denom() != 1 || v == w {
        //             continue;
        //         }
        //         let w_neigh : HashSet<V> = g.neighbor_vec(w).iter().cloned().collect();

        //         let pair = (v.min(w), v.max(w));
        //         if group_uncommon.contains_key(&pair) {
        //             continue;
        //         }
        //         group_uncommon.insert(pair, Self::uncommon_elements(&v_neigh, &w_neigh));
        //     }
        // }

        // for (_, u) in group_uncommon {
        //     for v in u.clone() {
        //         if !ncut_n.contains_key(&v) {
        //             continue;
        //         }
        //         let nu = u.len() as f64 - 2.0;
        //         // weights.entry(v).and_modify(|e| *e += 1.0 / nu);
        //         ncut_n.entry(v).and_modify(|e| *e += nu - 1.0);
        //         tts.entry(v).and_modify(|e| *e += 2);
        //         let ncut = *ncut_n.get(&v).unwrap();
        //         let rt = *tts.get(&v).unwrap();
        //         weights.insert(v, rt as f64 / ncut as f64);
        //     }
        // }

        for (k, _v) in weights.clone() {
            let ncut5 = *weights5.get(&k).unwrap();
            weights.entry(k).and_modify(|e| *e = f64::max(*e, (*e + 4.0 * ncut5) / (ncut5 + 1.0)));
        }

        let max_weight = weights.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let mut nbs = Vec::new();
        let mut frac = -1.0;
        if let Some((max_key, max_val)) = max_weight {
            if *max_val <= 1.0 {
                return (nbs, frac);
            }
            nbs = vec![*max_key];
            frac = 1.0 / max_val.clone() as f64;
        }

        (nbs, frac)
    }

    /// Split the graph into connected components
    /// Returns a vector of graphs
    pub fn split_components(g: &G) -> Vec<G> {
        let comps = g.component_vertices();
        comps.into_iter().map(|c| g.induced_subgraph(&c)).collect()
    }

    /// Returns a best occurrence of a cat state
    /// The first vertex in the result is the Clifford spider
    pub fn cat_ts(g: &G) -> Vec<V> {
        // the graph g is supposed to be completely simplified
        let prefered_order = [4, 6, 5, 3];
        let mut res = vec![];
        let mut index = None;

        for v in g.vertices() {
            if g.phase(v).denom() == &1 {
                let mut neigh = g.neighbor_vec(v);
                neigh.sort_by_key(|v| g.neighbor_vec(*v).len());
                if neigh.len() <= 6 {
                    if let Some(this_ind) = prefered_order.iter().position(|&r| r == neigh.len()) {
                        match index {
                            Some(ind) if this_ind < ind => {
                                res = vec![v];
                                res.append(&mut neigh);
                                index = Some(this_ind);
                            }
                            None => {
                                res = vec![v];
                                res.append(&mut neigh);
                                index = Some(this_ind);
                            }
                            _ => (),
                        }
                    }
                    if index == Some(0) {
                        break;
                    }
                }
            }
        }
        res
    }

    fn push_decomp(
        &mut self,
        fs: &[fn(&G, &[V]) -> G],
        depth: usize,
        g: &G,
        verts: &[V],
    ) -> &mut Self {
        for f in fs {
            let mut g = f(g, verts);
            match self.simp_func {
                FullSimp => {
                    crate::simplify::full_simp(&mut g);
                }
                CliffordSimp => {
                    crate::simplify::clifford_simp(&mut g);
                }
                _ => {}
            }

            // let comps = g.component_vertices();
            // if comps.len() > 1 {
            //     println!("GOT {} COMPONENTS ({})", comps.len(), comps.iter().map(|c| c.len()).format(","));
            // }
            self.stack.push_back((depth, g));
        }

        self
    }

    /// Perform the Bravyi-Smith-Smolin decomposition of 6 T gates
    /// into a sum of 7 terms
    ///
    /// See Section IV of:
    /// https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.021043
    ///
    /// In particular, see the text below equation (10) and
    /// equation (11) itself.
    ///
    fn push_bss_decomp(&mut self, depth: usize, g: &G, verts: &[V]) -> &mut Self {
        self.push_decomp(
            &[
                Decomposer::replace_b60,
                Decomposer::replace_b66,
                Decomposer::replace_e6,
                Decomposer::replace_o6,
                Decomposer::replace_k6,
                Decomposer::replace_phi1,
                Decomposer::replace_phi2,
            ],
            depth,
            g,
            verts,
        )
    }

    /// Perform a decomposition of 2 T gates in the symmetric 2-qubit
    /// space spanned by stabilisers
    fn push_sym_decomp(&mut self, depth: usize, g: &G, verts: &[V]) -> &mut Self {
        self.push_decomp(
            &[Decomposer::replace_bell_s, Decomposer::replace_epr],
            depth,
            g,
            verts,
        )
    }

    /// Replace a single T gate with its decomposition
    fn push_single_decomp(&mut self, depth: usize, g: &G, verts: &[V]) -> &mut Self {
        self.push_decomp(
            &[Decomposer::replace_t0, Decomposer::replace_t1],
            depth,
            g,
            verts,
        )
    }

    /// Perform a decomposition of 5 T-spiders, with one remaining
    fn push_magic5_from_cat_decomp(&mut self, depth: usize, g: &G, verts: &[V]) -> &mut Self {
        // println!("magic5");
        self.push_decomp(
            &[
                Decomposer::replace_magic5_0,
                Decomposer::replace_magic5_1,
                Decomposer::replace_magic5_2,
            ],
            depth,
            g,
            verts,
        )
    }

    /// Perform a decomposition of cat states
    fn push_cat_decomp(&mut self, depth: usize, g: &G, verts: &[V]) -> &mut Self {
        // verts[0] is a 0- or pi-spider, linked to all and only to vs in verts[1..] which are T-spiders
        let mut g = g.clone(); // that is annoying ...
        let mut verts = Vec::from(verts);
        if g.phase(verts[0]).numer() == &1 {
            g.set_phase(verts[0], Rational64::new(0, 1));
            let mut neigh = g.neighbor_vec(verts[1]);
            neigh.retain(|&x| x != verts[0]);
            for &v in &neigh {
                g.add_to_phase(v, Rational64::new(1, 1));
            }
            let tmp = g.phase(verts[1]);
            *g.scalar_mut() *= ScalarN::from_phase(tmp);
            g.set_phase(verts[1], g.phase(verts[1]) * Rational64::new(-1, 1));
        }
        if [3, 5].contains(&verts[1..].len()) {
            // if verts[1..].len() == 3 {
            //     println!("cat3 to 4");
            // } else {
            //     println!("cat5 to 6");
            // }
            let w = g.add_vertex(VType::Z);
            let v = g.add_vertex(VType::Z);
            g.add_edge_with_type(v, w, EType::H);
            g.add_edge_with_type(v, verts[0], EType::H);
            verts.push(v);
        }
        if verts[1..].len() == 6 {
            // println!("cat6");
            self.push_decomp(
                &[
                    Decomposer::replace_cat6_0,
                    Decomposer::replace_cat6_1,
                    Decomposer::replace_cat6_2,
                ],
                depth,
                &g,
                &verts,
            )
        } else if verts[1..].len() == 4 {
            // println!("cat4");
            self.push_decomp(
                &[Decomposer::replace_cat4_0, Decomposer::replace_cat4_1],
                // &[Decomposer::cut_cat4_0, Decomposer::cut_cat4_1],
                depth,
                &g,
                &verts,
            )
        } else {
            println!("this shouldn't be printed");
            self
        }
    }

    fn reverse_pivot(g: &mut G, vs0: &[V], vs1: &[V]) -> Vec<V> {
        let x = vs0.len() as i32;
        let y = vs1.len() as i32;
        g.scalar_mut().mul_sqrt2_pow(-(x - 1) * (y - 1));

        let v0 = g.add_vertex(VType::Z);
        let v1 = g.add_vertex(VType::Z);

        // Revert the edges between the neighbors of v0 and v1
        for &n0 in vs0 {
            for &n1 in vs1 {
                g.remove_edge(n0, n1);
            }
        }

        // Restore the original neighbors of v0 and v1
        for &n0 in vs0 {
            g.add_edge_smart(v0, n0, EType::H);
        }
        for &n1 in vs1 {
            g.add_edge_smart(v1, n1, EType::H);
        }

        g.add_edge_smart(v0, v1, EType::H);

        vec![v0, v1]
    }

    fn replace_tpair0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();

        let n = verts.len();

        let vs0 = Decomposer::reverse_pivot(&mut g, &verts[..n-2], &verts[n-2..]);

        Decomposer::replace_p0(&g, &vs0[..1])
    }

    fn replace_tpair1(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();

        let n = verts.len();

        let vs0 = Decomposer::reverse_pivot(&mut g, &verts[..n-2], &verts[n-2..]);

        Decomposer::replace_p1(&g, &vs0[..1])
    }

    fn cut_pg_0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();

        let x = g.add_vertex(VType::Z);

        for &v in &verts[2..] {
            g.remove_edge(v, verts[0]);
            g.add_edge_with_type(x, v, EType::H);
        }

        let w = g.add_vertex(VType::Z);
        g.add_edge_with_type(x, w, EType::H);
        g.add_edge_with_type(verts[0], w, EType::H);

        // Cut decomposition
        Decomposer::replace_p0(&g, &[w])
    }

    fn cut_pg_1(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();

        let x = g.add_vertex(VType::Z);

        for &v in &verts[2..] {
            g.remove_edge(v, verts[0]);
            g.add_edge_with_type(x, v, EType::H);
        }

        let w = g.add_vertex(VType::Z);
        g.add_edge_with_type(x, w, EType::H);
        g.add_edge_with_type(verts[0], w, EType::H);

        // Cut decomposition
        Decomposer::replace_p1(&g, &[w])
    }

    fn cut_cat4_0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();

        // Spider (un)fusing
        // g.remove_vertex(verts[0]);
        // let v0 = g.add_vertex(VType::Z);
        let v1 = g.add_vertex(VType::Z);

        g.remove_edge(verts[0], verts[3]);
        g.remove_edge(verts[0], verts[4]);
        // g.add_edge_with_type(verts[1], v0, EType::H);
        // g.add_edge_with_type(verts[2], v0, EType::H);
        g.add_edge_with_type(verts[3], v1, EType::H);
        g.add_edge_with_type(verts[4], v1, EType::H);

        // Identity spider
        let w = g.add_vertex(VType::Z);
        // g.add_edge_with_type(v0, w, EType::H);
        g.add_edge_with_type(v1, w, EType::H);
        g.add_edge_with_type(verts[0], w, EType::H);

        // Cut decomposition
        Decomposer::replace_p0(&g, &[w])
    }

    fn cut_cat4_1(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();

        // Spider (un)fusing
        // g.remove_vertex(verts[0]);
        // let v0 = g.add_vertex(VType::Z);
        let v1 = g.add_vertex(VType::Z);

        g.remove_edge(verts[0], verts[3]);
        g.remove_edge(verts[0], verts[4]);
        // g.add_edge_with_type(verts[1], v0, EType::H);
        // g.add_edge_with_type(verts[2], v0, EType::H);
        g.add_edge_with_type(verts[3], v1, EType::H);
        g.add_edge_with_type(verts[4], v1, EType::H);

        // Identity spider
        let w = g.add_vertex(VType::Z);
        // g.add_edge_with_type(v0, w, EType::H);
        g.add_edge_with_type(v1, w, EType::H);
        g.add_edge_with_type(verts[0], w, EType::H);

        // Cut decomposition
        Decomposer::replace_p1(&g, &[w])
    }

    // Basically the reverse of LC - basic_rules::local_comp_unchecked
    fn replace_sub_comp_0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        let p = Rational64::new(1, 2);

        let x = verts.len() as i32;
        g.scalar_mut().mul_sqrt2_pow(((x - 1) * (x - 2)) / 2);
        g.scalar_mut().mul_phase(Rational64::new(-*p.numer(), 4));

        // let v = g.add_vertex_with_phase(VType::Z, p);
        let v = g.add_vertex(VType::Z);

        for i in 0..verts.len() {
            g.add_to_phase(verts[i], p);
            g.add_edge_with_type(verts[i], v, EType::H);
            for j in (i + 1)..verts.len() {
                g.add_edge_smart(verts[i], verts[j], EType::H);
            }
        }

        *g.scalar_mut() *= ScalarN::Exact(-1, vec![0, 1, 0, -1]);
        let w = g.add_vertex(VType::Z);
        g.add_edge_with_type(v, w, EType::H);
        // g.add_to_phase(v, Rational64::new(-1, 4));

        g
    }

    fn replace_sub_comp_1(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        let p = Rational64::new(1, 2);

        let x = verts.len() as i32;
        g.scalar_mut().mul_sqrt2_pow(((x - 1) * (x - 2)) / 2);
        g.scalar_mut().mul_phase(Rational64::new(-*p.numer(), 4));

        // let v = g.add_vertex_with_phase(VType::Z, p);
        let v = g.add_vertex(VType::Z);

        for i in 0..verts.len() {
            g.add_to_phase(verts[i], p);
            g.add_edge_with_type(verts[i], v, EType::H);
            for j in (i + 1)..verts.len() {
                g.add_edge_smart(verts[i], verts[j], EType::H);
            }
        }

        // i/sqrt{2}
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![0, 1, 0, 1]);
        let w = g.add_vertex_with_phase(VType::Z, Rational64::one());
        g.add_edge_with_type(v, w, EType::H);
        g
    }

    fn replace_cat6_0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![1, 0, 0, 0]);
        for &v in &verts[1..] {
            g.add_to_phase(v, Rational64::new(-1, 4));
            g.set_edge_type(v, verts[0], EType::N);
        }
        g.set_phase(verts[0], Rational64::new(-1, 2));
        g
    }

    fn replace_cat6_1(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![-1, 0, 1, 0]);
        for &v in &verts[1..] {
            g.add_to_phase(v, Rational64::new(-1, 4));
        }
        g
    }

    fn replace_cat6_2(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(7, vec![0, -1, 0, 0]);
        for i in 1..verts.len() {
            g.add_to_phase(verts[i], Rational64::new(-1, 4));
            for j in i + 1..verts.len() {
                g.add_edge_smart(verts[i], verts[j], EType::H);
            }
        }
        g
    }

    fn replace_magic5_0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(1, vec![1, 0, 0, 0]);
        for &v in verts {
            g.add_to_phase(v, Rational64::new(-1, 4));
            g.add_edge_smart(v, verts[0], EType::N);
        }
        g.add_to_phase(verts[0], Rational64::new(-3, 4));
        g
    }

    fn replace_magic5_1(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(1, vec![-1, 0, 1, 0]);
        let p = g.add_vertex(VType::Z);
        for &v in verts {
            g.add_to_phase(v, Rational64::new(-1, 4));
            g.add_edge_with_type(v, p, EType::H);
        }
        let w = g.add_vertex_with_phase(VType::Z, Rational64::new(-1, 4));
        g.add_edge_with_type(w, p, EType::H);
        g
    }

    fn replace_magic5_2(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(9, vec![0, -1, 0, 0]);
        let p = g.add_vertex(VType::Z);
        let w = g.add_vertex_with_phase(VType::Z, Rational64::new(-1, 4));
        g.add_edge_with_type(p, w, EType::H);
        for i in 0..verts.len() {
            g.add_to_phase(verts[i], Rational64::new(-1, 4));
            g.add_edge_with_type(verts[i], p, EType::H);
            g.add_edge_with_type(verts[i], w, EType::H);
            for j in i + 1..verts.len() {
                g.add_edge_smart(verts[i], verts[j], EType::H);
            }
        }
        g
    }

    fn replace_cat4_0(g: &G, verts: &[V]) -> G {
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(0, vec![0, 0, 1, 0]);
        for &v in &verts[1..] {
            g.add_to_phase(v, Rational64::new(-1, 4));
        }
        g
    }

    fn replace_cat4_1(g: &G, verts: &[V]) -> G {
        // same as replace_cat6_0, only with a different scalar
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![1, 0, -1, 0]);
        for &v in &verts[1..] {
            g.add_to_phase(v, Rational64::new(-1, 4));
            g.set_edge_type(v, verts[0], EType::N);
        }
        g.set_phase(verts[0], Rational64::new(-1, 2));
        g
    }

    fn replace_b60(g: &G, verts: &[V]) -> G {
        // println!("replace_b60");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-2, vec![-1, 0, 1, 1]);
        for &v in &verts[0..6] {
            g.add_to_phase(v, Rational64::new(-1, 4));
        }
        g
    }

    fn replace_b66(g: &G, verts: &[V]) -> G {
        // println!("replace_b66");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-2, vec![-1, 0, 1, -1]);
        for &v in verts {
            g.add_to_phase(v, Rational64::new(3, 4));
        }
        g
    }

    fn replace_e6(g: &G, verts: &[V]) -> G {
        // println!("replace_e6");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(1, vec![0, -1, 0, 0]);

        let w = g.add_vertex_with_phase(VType::Z, Rational64::one());
        for &v in verts {
            g.add_to_phase(v, Rational64::new(1, 4));
            g.add_edge_with_type(v, w, EType::H);
        }

        g
    }

    fn replace_o6(g: &G, verts: &[V]) -> G {
        // println!("replace_o6");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(1, vec![-1, 0, -1, 0]);

        let w = g.add_vertex(VType::Z);
        for &v in verts {
            g.add_to_phase(v, Rational64::new(1, 4));
            g.add_edge_with_type(v, w, EType::H);
        }

        g
    }

    fn replace_k6(g: &G, verts: &[V]) -> G {
        // println!("replace_k6");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(1, vec![1, 0, 0, 0]);

        let w = g.add_vertex_with_phase(VType::Z, Rational64::new(-1, 2));
        for &v in verts {
            g.add_to_phase(v, Rational64::new(-1, 4));
            g.add_edge_with_type(v, w, EType::N);
        }

        g
    }

    fn replace_phi1(g: &G, verts: &[V]) -> G {
        // println!("replace_phi1");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(3, vec![1, 0, 1, 0]);

        let mut ws = vec![];
        for i in 0..5 {
            let w = g.add_vertex(VType::Z);
            ws.push(w);
            g.add_edge_with_type(verts[i], ws[i], EType::H);
            g.add_edge_with_type(ws[i], verts[5], EType::H);
            g.add_to_phase(verts[i], Rational64::new(-1, 4));
        }

        g.add_to_phase(verts[5], Rational64::new(3, 4));

        g.add_edge_with_type(ws[0], ws[2], EType::H);
        g.add_edge_with_type(ws[0], ws[3], EType::H);
        g.add_edge_with_type(ws[1], ws[3], EType::H);
        g.add_edge_with_type(ws[1], ws[4], EType::H);
        g.add_edge_with_type(ws[2], ws[4], EType::H);

        g
    }

    fn replace_phi2(g: &G, verts: &[V]) -> G {
        // print!("replace_phi2 -> ");
        Decomposer::replace_phi1(
            g,
            &[verts[0], verts[1], verts[3], verts[4], verts[5], verts[2]],
        )
    }

    fn replace_bell_s(g: &G, verts: &[V]) -> G {
        // println!("replace_bell_s");
        let mut g = g.clone();
        g.add_edge_smart(verts[0], verts[1], EType::N);
        g.add_to_phase(verts[0], Rational64::new(-1, 4));
        g.add_to_phase(verts[1], Rational64::new(1, 4));

        g
    }

    fn replace_epr(g: &G, verts: &[V]) -> G {
        // println!("replace_epr");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::from_phase(Rational64::new(1, 4));
        let w = g.add_vertex_with_phase(VType::Z, Rational64::one());
        for &v in verts {
            g.add_edge_with_type(v, w, EType::H);
            g.add_to_phase(v, Rational64::new(-1, 4));
        }

        g
    }

    fn replace_t0(g: &G, verts: &[V]) -> G {
        // println!("replace_t0");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![0, 1, 0, -1]);
        let w = g.add_vertex(VType::Z);
        g.add_edge_with_type(verts[0], w, EType::H);
        g.add_to_phase(verts[0], Rational64::new(-1, 4));
        g
    }

    fn replace_t1(g: &G, verts: &[V]) -> G {
        // println!("replace_t1");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![1, 0, 1, 0]);
        let w = g.add_vertex_with_phase(VType::Z, Rational64::one());
        g.add_edge_with_type(verts[0], w, EType::H);
        g.add_to_phase(verts[0], Rational64::new(-1, 4));
        g
    }

    // Cut 0-spider
    fn replace_p0(g: &G, verts: &[V]) -> G {
        // println!("replace_p0");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![0, 1, 0, -1]);
        let w = g.add_vertex(VType::Z);
        g.add_edge_with_type(verts[0], w, EType::H);
        g
    }

    fn replace_p1(g: &G, verts: &[V]) -> G {
        // println!("replace_p1");
        let mut g = g.clone();
        *g.scalar_mut() *= ScalarN::Exact(-1, vec![0, 1, 0, -1]);
        let w = g.add_vertex_with_phase(VType::Z, Rational64::one());
        g.add_edge_with_type(verts[0], w, EType::H);
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;
    use crate::vec_graph::Graph;

    #[test]
    fn bss_scalars() {
        // this test is mainly to record how each of the exact
        // form scalars for the BSS decomposition were computed
        let one = ScalarN::one();
        let om = ScalarN::Exact(0, vec![0, 1, 0, 0]);
        let om2 = &om * &om;
        let om7 = ScalarN::Exact(0, vec![0, 0, 0, -1]);
        assert_eq!(&om * &om7, ScalarN::one());

        let minus = ScalarN::Exact(0, vec![-1, 0, 0, 0]);
        let onefourth = ScalarN::Exact(-2, vec![1, 0, 0, 0]);
        let two = &one + &one;
        let sqrt2 = ScalarN::sqrt2();
        let eight = &two * &two * &two;

        let k6 = &om7 * &two * &om;
        let phi = &om7 * &eight * &sqrt2 * &om2;
        let b60 = &om7 * &minus * &onefourth * (&one + &sqrt2);
        let b66 = &om7 * &onefourth * (&one + (&minus * &sqrt2));
        let o6 = &om7 * &minus * &two * &sqrt2 * &om2;
        let e6 = &om7 * &minus * &two * &om2;

        assert_eq!(b60, ScalarN::Exact(-2, vec![-1, 0, 1, 1]));
        assert_eq!(b66, ScalarN::Exact(-2, vec![-1, 0, 1, -1]));
        assert_eq!(e6, ScalarN::Exact(1, vec![0, -1, 0, 0]));
        assert_eq!(o6, ScalarN::Exact(1, vec![-1, 0, -1, 0]));
        assert_eq!(k6, ScalarN::Exact(1, vec![1, 0, 0, 0]));
        assert_eq!(phi, ScalarN::Exact(3, vec![1, 0, 1, 0]));
    }

    #[test]
    fn single_scalars() {
        let s0 = ScalarN::sqrt2_pow(-1);
        let s1 = ScalarN::from_phase(Rational64::new(1, 4)) * &s0;
        println!("s0 = {:?}\ns1 = {:?}", s0, s1);
        assert_eq!(s0, ScalarN::Exact(-1, vec![0, 1, 0, -1]));
        assert_eq!(s1, ScalarN::Exact(-1, vec![1, 0, 1, 0]));
    }

    #[test]
    fn single() {
        let mut g = Graph::new();
        let v = g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
        let w = g.add_vertex(VType::B);
        g.add_edge(v, w);
        g.set_outputs(vec![w]);

        let mut d = Decomposer::new(&g);
        d.decomp_top();
        assert_eq!(d.stack.len(), 2);

        let t = g.to_tensor4();
        let mut tsum = Tensor4::zeros(vec![2]);
        for (_, h) in &d.stack {
            tsum = tsum + h.to_tensor4();
        }
        assert_eq!(t, tsum);
    }

    #[test]
    fn sym() {
        let mut g = Graph::new();
        let mut outs = vec![];
        for _ in 0..2 {
            let v = g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
            let w = g.add_vertex(VType::B);
            outs.push(w);
            g.add_edge(v, w);
        }
        g.set_outputs(outs);

        let mut d = Decomposer::new(&g);
        d.decomp_top();
        assert_eq!(d.stack.len(), 2);

        let t = g.to_tensor4();
        let mut tsum = Tensor4::zeros(vec![2; 2]);
        for (_, h) in &d.stack {
            tsum = tsum + h.to_tensor4();
        }
        assert_eq!(t, tsum);
    }

    #[test]
    fn bss() {
        let mut g = Graph::new();
        let mut outs = vec![];
        for _ in 0..6 {
            let v = g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
            let w = g.add_vertex(VType::B);
            outs.push(w);
            g.add_edge(v, w);
        }
        g.set_outputs(outs);

        let mut d = Decomposer::new(&g);
        d.decomp_top();
        assert_eq!(d.stack.len(), 7);

        let t = g.to_tensor4();
        let mut tsum = Tensor4::zeros(vec![2; 6]);
        for (_, h) in &d.stack {
            tsum = tsum + h.to_tensor4();
        }
        assert_eq!(t, tsum);
    }

    #[test]
    fn mixed() {
        let mut g = Graph::new();
        let mut outs = vec![];
        for _ in 0..9 {
            let v = g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
            let w = g.add_vertex(VType::B);
            outs.push(w);
            g.add_edge(v, w);
        }
        g.set_outputs(outs);

        let mut d = Decomposer::new(&g);
        d.save(true);
        assert_eq!(d.max_terms(), 7.0 * 2.0 * 2.0);
        while !d.stack.is_empty() {
            d.decomp_top();
        }

        assert_eq!(d.done.len(), 7 * 2 * 2);

        // thorough but SLOW
        // let t = g.to_tensor4();
        // let mut tsum = Tensor4::zeros(vec![2; 9]);
        // for h in &d.done { tsum = tsum + h.to_tensor4(); }
        // assert_eq!(t, tsum);
    }

    #[test]
    fn mixed_sc() {
        let mut g = Graph::new();
        for i in 0..11 {
            g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));

            for j in 0..i {
                g.add_edge_with_type(i, j, EType::H);
            }
            // let w = g.add_vertex(VType::Z);
            // g.add_edge(v, w);
        }

        let mut d = Decomposer::new(&g);
        d.with_full_simp();
        // assert_eq!(d.max_terms(), 7.0*2.0*2.0);
        d.decomp_all();
        // assert_eq!(d.nterms, 7*2*2);

        let sc = g.to_tensor4()[[]];
        assert_eq!(Scalar::from_scalar(&sc), d.scalar);
    }

    #[test]
    fn all_and_depth() {
        let mut g = Graph::new();
        let mut outs = vec![];
        for _ in 0..9 {
            let v = g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
            let w = g.add_vertex(VType::B);
            outs.push(w);
            g.add_edge(v, w);
        }
        g.set_outputs(outs);

        let mut d = Decomposer::new(&g);
        d.with_full_simp();
        d.save(true).decomp_all();
        assert_eq!(d.done.len(), 7 * 2 * 2);
        let mut d = Decomposer::new(&g);
        d.with_full_simp();
        d.decomp_until_depth(2);
        assert_eq!(d.stack.len(), 7 * 2);
    }

    #[test]
    fn full_simp() {
        let mut g = Graph::new();
        let mut outs = vec![];
        for _ in 0..9 {
            let v = g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
            let w = g.add_vertex(VType::B);
            outs.push(w);
            g.add_edge(v, w);
        }
        g.set_outputs(outs);

        let mut d = Decomposer::new(&g);
        d.with_full_simp().save(true).decomp_all();
        assert_eq!(d.done.len(), 7 * 2 * 2);
    }

    #[test]
    fn test_disconnected_vertices() {
        let mut g = Graph::new();

        // 6 disconnected vertices
        for _ in 0..6 {
            g.add_vertex_with_phase(VType::Z, Rational64::new(1, 4));
        }

        let mut d = Decomposer::new(&g);
        d.with_full_simp();
        d.decomp_all();

        // 7 terms from doing BSS
        assert_eq!(d.nterms, 7);

        let sc = g.to_tensor4()[[]];
        assert_eq!(Scalar::from_scalar(&sc), d.scalar);
    }

    #[test]
    fn test_disconnected_cats() {
        let mut g = Graph::new();

        let ns = [4, 6, 5, 3];
        // let mut all_bs: Vec<V> = Vec::new();

        for n in ns {
            let catn = g.add_cat(n);
        }

        let mut d = Decomposer::new(&g);
        d.use_cats(true);
        d.with_full_simp();
        d.decomp_all();

        // terms from doing cat decomp
        assert_eq!(d.nterms, 2 * 3 * 3 * 2);

        let sc = g.to_tensor4()[[]];
        assert_eq!(Scalar::from_scalar(&sc), d.scalar);

        let mut d2 = Decomposer::new(&g);
        d2.use_cats(true);
        d2.with_full_simp();
        d2.split_comps(true);
        d2.decomp_all();

        // terms from doing cat decomp
        assert_eq!(d2.nterms, 2 + 3 + 3 + 2);

        let sc = g.to_tensor4()[[]];
        assert_eq!(Scalar::from_scalar(&sc), d2.scalar);
    }

    // #[test]
    // fn test_split_comp() {
    //     use crate::circuit::Circuit;
    //     let c = Circuit::random_pauli_gadget()
    //         .qubits(10)
    //         .depth(5)
    //         .seed(1337)
    //         .min_weight(2)
    //         .max_weight(4)
    //         .build();

    //     // let mut rng = StdRng::seed_from_u64(1337 * 37);

    //     let mut g: Graph = c.to_graph();
    //     crate::simplify::full_simp(&mut g);
    //     // let n = g.num_vertices();
    //     let g2 = g.clone();
    //     let g3 = g.clone();
    //     g.append_graph(&g2);
    //     g.append_graph(&g3);
    //     let comps = g.component_vertices();
    //     // let mut g = Graph::new();
    //     assert_eq!(comps.len(), 3);

    //     let cat3 = g.add_cat(3);
    //     let mut i = 0;
    //     for v in g.vertices().collect::<Vec<_>>() {
    //         if comps[i].contains(&v) {
    //             g.add_edge(cat3[i+1], v);
    //             i += 1;
    //         }
    //         if i == 3 {
    //             break;
    //         }
    //     }

    //     let mut d = Decomposer::new(&g);
    //     d.use_cats(true);
    //     d.with_full_simp();
    //     let d = d.decomp_parallel(3);

    //     // terms from doing cat decomp
    //     println!("nterms = {}", d.nterms);
    //     // assert_eq!(d.nterms, 12);

    //     // let sc = g.to_tensor4()[[]];

    //     // assert_eq!(Scalar::from_scalar(&sc), d.scalar);

    //     let mut d2 = Decomposer::new(&g);
    //     d2.use_cats(true);
    //     d2.with_full_simp();
    //     d2.split_comps(true);
    //     d2.decomp_all();

    //     // terms from doing cat decomp
    //     // assert_eq!(d2.nterms, 10);

    //     println!("nterms = {}", d2.nterms);

    //     // let sc = g.to_tensor4()[[]];
    //     // assert_eq!(Scalar::from_scalar(&sc), d2.scalar);

    // }
}
