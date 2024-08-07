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

use itertools::Itertools;
use quizx::circuit::*;
use quizx::decompose::Decomposer;
use quizx::graph::*;
use quizx::scalar::*;
use quizx::vec_graph::Graph;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let debug = true;
    let use_comp = true;
    let use_heur = false;
    let use_paired_heur = false; 
    let use_sub_comp = false;
    let args: Vec<_> = env::args().collect();
    let (qs, n_ccz, seed) = if args.len() >= 4 {
        (
            args[1].parse().unwrap(),
            args[2].parse().unwrap(),
            args[3].parse().unwrap(),
        )
    } else {
        (50, 30, 1337)
    };
    if debug {
        println!("qubits: {}, # ccz: {}, seed: {}", qs, n_ccz, seed);
    }

    // generate hidden shift circuit as in Bravyi-Gosset 2016
    let (c, shift) = Circuit::random_hidden_shift()
        .qubits(qs)
        .n_ccz(n_ccz) // T = CCZ * 2 * 7
        .seed((seed * qs * n_ccz) as u64)
        .build();

    // compute T-count and theoretical max terms
    // let mut g: Graph = c.to_graph();
    // let tcount = g.tcount();
    // g.plug(&g.to_adjoint());
    // let mut d = Decomposer::new(&g);
    // let naive = d.max_terms();

    let time_all = Instant::now();
    // let mut shift_m = vec![];
    // let mut terms = 0;
    // let mut tcounts = vec![];

    // Hidden shift is deterministic ==> only need 1-qubit marginals

    // compute marginals P(qi = 1)
    // for i in 0..qs {
    let mut g: Graph = c.to_graph();

    // |00..0> as input
    g.plug_inputs(&vec![BasisElem::Z0; qs]);

    // <1|_qi ⊗ I as output
    let mut rng = StdRng::seed_from_u64(seed as u64 * 37);
    // g.plug_output(i, BasisElem::Z1);

    // // compute norm as <psi|psi>. Doubles T-count!
    // g.plug(&g.to_adjoint());
    let effect: Vec<_> = (0..qs)
        .map(|_| {
            if rng.gen_bool(0.5) {
                BasisElem::Z0
            } else {
                BasisElem::Z1
            }
        })
        .collect();

    let mut h = g.clone();
    h.plug_outputs(&effect);

    quizx::simplify::full_simp(&mut h);
    // tcounts.push(g.tcount());
    let tcount = h.tcount();

    if debug {
        print!("initial ({}T), reduced ({}T):\t", g.tcount(), tcount);
    }
    io::stdout().flush().unwrap();

    let time = Instant::now();

    // do the decomposition, with full_simp called eagerly
    let mut d = Decomposer::new(&h);
    d.use_cats(true);
    d.split_comps(use_comp);
    d.use_heur(use_heur);
    d.use_paired_heur(use_paired_heur);
    d.use_sub_comp(use_sub_comp);
    d.with_full_simp();

    let naive = d.max_terms();
    let d = d.decomp_parallel(3);
    // let d = d.decomp_all();
    // terms += d.nterms;

    let alpha = if tcount > 0 { (d.nterms as f64).log2() / (tcount as f64) } else { -1.0 };

    // record the measurement outcome. Since hidden shift is deterministic, we
    // only need to check if the marginal P(q_i = 1) is zero for each i.
    let outcome = if d.scalar.is_zero() { 0 } else { 1 };
    // shift_m.push(outcome);

    if debug {
        println!(
            "{} (terms: {}, time: {:.2?})",
            outcome,
            d.nterms,
            time.elapsed()
        );
    }
    // }

    let time = time_all.elapsed();
    if debug {
        println!("Shift: {}", shift.iter().format(""));
        // println!("Simul: {}", shift_m.iter().format(""));
        // println!("Check: {}", shift == shift_m);
        println!(
            "Circuit with {} qubits and T-count {} simulated in {:.2?}",
            qs, tcount, time
        );
        println!("Got {} terms, alpha = {:.3} ({:+e} naive)", d.nterms, alpha, naive);
    }

    let data = format!(
        "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
        qs,
        n_ccz,
        seed,
        d.nterms,
        time.as_millis(),
        tcount,
        alpha
    );
    // if shift == shift_m {
    print!("OK {}", data);
    fs::write(format!("hidden_shift_{}_{}_{}", qs, n_ccz, seed), data)
        .expect("Unable to write file");
    // } else {
    //     print!("FAILED {}", data);
    // }
    Ok(())
}
