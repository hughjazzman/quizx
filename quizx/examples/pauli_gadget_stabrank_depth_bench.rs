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
use quizx::decompose::{terms_for_tcount, Decomposer};
use quizx::graph::*;
use quizx::scalar::*;
use quizx::tensor::*;
use quizx::vec_graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
// use std::fs;
// use std::io::{self, Write};
// use std::time::{Duration, Instant};

fn meas_str(e: &[BasisElem]) -> String {
    format!(
        "{}",
        e.iter()
            .map(|&b| if b == BasisElem::Z0 { 0 } else { 1 })
            .format("")
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let debug = true;
    let use_heur = true;
    let args: Vec<_> = env::args().collect();
    let (qs, depth, min_weight, max_weight, seed) = if args.len() >= 5 {
        (
            args[1].parse().unwrap(),
            args[2].parse().unwrap(),
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
        )
    } else {
        (50, 70, 4, 4, 1337)
        // (13, 15, 2, 4, 3, 1337)
    };
    // if debug {
    //     println!(
    //         "qubits: {}, depth: {}, min_weight: {}, max_weight: {}, seed: {}",
    //         qs, depth, min_weight, max_weight, seed
    //     );
    // }

    // let time_all = Instant::now();

    let mut depths = vec![];
    let mut tcounts = vec![];
    let mut red_tcounts = vec![];
    let mut terms = vec![];
    // let mut success = true;
    // let mut time = Duration::from_millis(0);
    // let mut times = vec![];
    let mut alphas = vec![];
    // let mut max_alpha: f64 = -1.0;

    // for depth in (1..60).step_by(3) {
        depths.push(depth);
        // Get random circuit
        let c = Circuit::random_pauli_gadget()
            .qubits(qs)
            .depth(depth)
            .seed(seed)
            .min_weight(min_weight)
            .max_weight(max_weight)
            .build();


        // let c = Circuit::random()
        //     .qubits(qs)
        //     .depth(depth * 10)
        //     .seed(seed)
        //     // .clifford_t(0.25)
        //     .p_cnot(0.4)
        //     .p_t(0.4)
        //     .p_cz(0.2/3.0)
        //     .p_h(0.2/3.0)
        //     .p_s(0.2/3.0)
        //     .build();

        let mut rng = StdRng::seed_from_u64(seed * 37);

        let mut g: Graph = c.to_graph();

        // Plug inputs and outputs
        g.plug_inputs(&vec![BasisElem::Z0; qs]);

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

        let tcount = h.tcount();
        tcounts.push(tcount);

        // if debug {
        //     println!("g has T-count: {}", tcount);
        // }

        // ZX-simplify
        quizx::simplify::full_simp(&mut h);

        let red_tcount = h.tcount();
        red_tcounts.push(red_tcount);

        // if debug {
        //     println!("g has reduced T-count: {}", red_tcount);
        // }

        // if debug {
        //     println!("Plugging {} ({}T)", meas_str(&effect), red_tcount);
        // }

        // Stabiliser decomposition
        let mut d = Decomposer::new(&h);
        d.use_cats(true);
        d.split_comps(true);
        d.use_heur(use_heur);
        d.with_full_simp();

        let d = d.decomp_parallel(3);

        // Stabiliser decomposition
        let mut d2 = Decomposer::new(&h);
        d2.use_cats(true);
        d2.split_comps(true);
        d2.use_heur(use_heur);
        d2.is_bench(true);
        d2.with_full_simp();

        let d2 = d2.decomp_parallel(3);

        // let prob = &d.scalar * &d.scalar.conj();
        terms.push(d.nterms);

        let mut c_alpha = -1.0;
        let mut c_alpha2 = -1.0;
        if red_tcount > 0 {
            c_alpha = (d.nterms as f64).log2() / (red_tcount as f64);
            c_alpha2 = (d2.nterms as f64).log2() / (red_tcount as f64);
        }
        // max_alpha = max_alpha.max(c_alpha);
        alphas.push(c_alpha);

        if debug && red_tcount > 0 && d.nterms > d2.nterms {
            println!("worse");
            println!("d1: {:?}, d2: {:?}", d.nterms, d2.nterms);
            println!("tcount: {:?}", red_tcount);
            println!("alpha1: {:?}, alpha2: {:?}", c_alpha, c_alpha2);
            println!("{}", h.to_dot());
        } else if debug && red_tcount > 0 {
            // println!("better");
            // println!("d1: {:?}, d2: {:?}", d.nterms, d2.nterms);
            // println!("tcount: {:?}", red_tcount);
            // println!("alpha1: {:?}, alpha2: {:?}", c_alpha, c_alpha2);
            // println!("{}", h.to_dot());
        }

    //     if debug {
    //         println!(
    //             "terms = {}, a = {:.3}, P = {}, re(P) ~ {}",
    //             d.nterms,
    //             c_alpha,
    //             prob,
    //             prob.float_value().re
    //         );
    //     }

    //     times.push(time_all.elapsed().as_millis());
    //     time += time_all.elapsed();

    //     // for small numbers of qubits, it is feasible to check the final probablility
    //     success = success
    //         && if qs <= 15 {
    //             print!("Checking tensor...");
    //             io::stdout().flush().unwrap();
    //             let mut check: Graph = c.to_graph();
    //             check.plug_inputs(&vec![BasisElem::Z0; qs]);
    //             check.plug_outputs(&effect);
    //             let amp = check.to_tensor4()[[]];
    //             let check_prob = amp * amp.conj();
    //             if Scalar::from_scalar(&check_prob) == prob {
    //                 println!("OK");
    //                 true
    //             } else {
    //                 println!("FAILED {} != {}", check_prob, prob);
    //                 false
    //             }
    //         } else {
    //             println!("Skipping tensor check (too big).");
    //             true
    //         };

    //     if debug {
    //         println!(
    //             "Circuit with {} qubits and T-count {} (red-T {}) simulated in {:.2?}",
    //             qs, tcount, red_tcount, time
    //         );
    //     }
    // // }

    // // let naive: f64 = (nsamples as f64) * (qs as f64) * terms_for_tcount(2 * tcount);
    // // let no_simp: f64 = tcounts.iter().map(|&t| terms_for_tcount(t)).sum();
    // // println!(
    // //     "Got {:.3} max alpha across all depths ({} max T-count)",
    // //     max_alpha,
    // //     red_tcounts.iter().max().unwrap()
    // // );

    // // let data = format!(
    // //     "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
    // //     qs,
    // //     depths.iter().format(","),
    // //     tcounts.iter().format(","),
    // //     red_tcounts.iter().format(","),
    // //     min_weight,
    // //     max_weight,
    // //     seed,
    // //     terms.iter().format(","),
    // //     times.iter().format(","),
    // //     alphas.iter().format(",")
    // // );    
    // let data = format!(
    //     "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{:.6}\"\n",
    //     qs,
    //     depth,
    //     tcount,
    //     red_tcount,
    //     min_weight,
    //     max_weight,
    //     seed,
    //     d.nterms,
    //     time.as_millis(),
    //     c_alpha
    // );
    // if success {
    //     print!("OK {}\n", data);
    // } else {
    //     print!("FAILED {}", data);
    // }
    // fs::write(
    //     format!("pauli_gadget_{}_{}_{}_{}_{}", qs, depth, min_weight, max_weight, seed),
    //     data,
    // )
    // .expect("Unable to write file");

    Ok(())
}
