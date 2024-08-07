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
use std::fs;
use std::io::{self, Write};
use std::time::{Duration, Instant};

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
    let use_paired_heur = true;
    let use_comp = true;
    let use_sub_comp = false;
    let args: Vec<_> = env::args().collect();
    let (qs, depth,  seed) = if args.len() >= 3 {
        (
            args[1].parse().unwrap(),
            args[2].parse().unwrap(),
            args[3].parse().unwrap(),
        )
    } else {
        (50, 70, 1337)
        // (13, 15, 2, 4, 3, 1337)
    };
    if debug {
        println!(
            "qubits: {}, depth: {}, seed: {}",
            qs, depth, seed
        );
    }

    let time_all = Instant::now();

    let mut depths = vec![];
    let mut tcounts = vec![];
    let mut red_tcounts = vec![];
    let mut terms = vec![];
    let mut success = true;
    let mut time = Duration::from_millis(0);
    let mut times = vec![];
    let mut alphas = vec![];
    // let mut max_alpha: f64 = -1.0;

    let depth = if qs > 50 { (depth + 200) >> 1 } else { depth };

    // for depth in (1..60).step_by(3) {
    depths.push(depth);
    // Get random circuit
    let c = Circuit::random_ccz()
        .qubits(qs)
        .depth(depth * 10)
        .seed(seed)
        .p_ccz(0.05)
        .p_t(0.05)
        .with_cliffords()
        .build();

    // let c = Circuit::random()
    //     .qubits(qs)
    //     .depth(depth * 10)
    //     .seed(seed)
    //     // .clifford_t(0.25)
    //     .p_cnot(0.4)
    //     .p_t(0.4)
    //     .p_cz(0.2 / 3.0)
    //     .p_h(0.2 / 3.0)
    //     .p_s(0.2 / 3.0)
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

    if debug {
        println!("g has T-count: {}", tcount);
    }

    // ZX-simplify
    quizx::simplify::full_simp(&mut h);

    let red_tcount = h.tcount();
    red_tcounts.push(red_tcount);

    if debug {
        println!("g has reduced T-count: {}", red_tcount);
    }

    if debug {
        println!("Plugging {} ({}T)", meas_str(&effect), red_tcount);
    }

    // Stabiliser decomposition
    let mut d = Decomposer::new(&h);
    d.use_cats(true);
    d.split_comps(use_comp);
    d.use_heur(use_heur);
    d.use_paired_heur(use_paired_heur);
    d.use_sub_comp(use_sub_comp);
    d.with_full_simp();

    let d = d.decomp_parallel(3);
    let prob = &d.scalar * &d.scalar.conj();
    terms.push(d.nterms);

    let mut c_alpha = -1.0;
    if red_tcount > 0 {
        c_alpha = (d.nterms as f64).log2() / (red_tcount as f64);
    }
    // max_alpha = max_alpha.max(c_alpha);
    alphas.push(c_alpha);

    if debug {
        println!(
            "terms = {}, a = {:.3}, P = {}, re(P) ~ {}",
            d.nterms,
            c_alpha,
            prob,
            prob.float_value().re
        );
    }

    times.push(time_all.elapsed().as_millis());
    time += time_all.elapsed();

    // for small numbers of qubits, it is feasible to check the final probablility
    success = success
        && if qs <= 15 {
            print!("Checking tensor...");
            io::stdout().flush().unwrap();
            let mut check: Graph = c.to_graph();
            check.plug_inputs(&vec![BasisElem::Z0; qs]);
            check.plug_outputs(&effect);
            let amp = check.to_tensor4()[[]];
            let check_prob = amp * amp.conj();
            if Scalar::from_scalar(&check_prob) == prob {
                println!("OK");
                true
            } else {
                println!("FAILED {} != {}", check_prob, prob);
                false
            }
        } else {
            println!("Skipping tensor check (too big).");
            true
        };

    if debug {
        println!(
            "Circuit with {} qubits and T-count {} (red-T {}) simulated in {:.2?}",
            qs, tcount, red_tcount, time
        );
    }
    // }

    // let naive: f64 = (nsamples as f64) * (qs as f64) * terms_for_tcount(2 * tcount);
    // let no_simp: f64 = tcounts.iter().map(|&t| terms_for_tcount(t)).sum();
    // println!(
    //     "Got {:.3} max alpha across all depths ({} max T-count)",
    //     max_alpha,
    //     red_tcounts.iter().max().unwrap()
    // );

    // let data = format!(
    //     "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
    //     qs,
    //     depths.iter().format(","),
    //     tcounts.iter().format(","),
    //     red_tcounts.iter().format(","),
    //     min_weight,
    //     max_weight,
    //     seed,
    //     terms.iter().format(","),
    //     times.iter().format(","),
    //     alphas.iter().format(",")
    // );
    let data = format!(
        "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{:.6}\"\n",
        qs,
        depth,
        tcount,
        red_tcount,
        seed,
        d.nterms,
        time.as_millis(),
        c_alpha
    );
    if success {
        print!("OK {}\n", data);
    } else {
        print!("FAILED {}", data);
    }
    fs::write(
        format!(
            "rand_ccz_{}_{}_{}",
            qs, depth, seed
        ),
        data,
    )
    .expect("Unable to write file");

    Ok(())
}
