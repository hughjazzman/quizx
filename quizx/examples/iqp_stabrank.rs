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
    let args: Vec<_> = env::args().collect();
    let (qs, seed) = if args.len() >= 2 {
        (args[1].parse().unwrap(), args[2].parse().unwrap())
    } else {
        (50, 1337)
    };

    if debug {
        println!("qubits: {}, seed: {}", qs, seed);
    }

    let time_all = Instant::now();

    let mut success = true;
    let mut time = Duration::from_millis(0);

    let c = Circuit::random_iqp().qubits(qs).build();

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

    if debug {
        println!("g has T-count: {}", tcount);
    }

    // ZX-simplify
    quizx::simplify::full_simp(&mut h);

    let red_tcount = h.tcount();

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
    d.with_full_simp();

    let d = d.decomp_parallel(3);
    let prob = &d.scalar * &d.scalar.conj();

    let c_alpha = if red_tcount > 0 {
        (d.nterms as f64).log2() / (red_tcount as f64)
    } else {
        -1.0
    };

    if debug {
        println!(
            "terms = {}, a = {:.3}, P = {}, re(P) ~ {}",
            d.nterms,
            c_alpha,
            prob,
            prob.float_value().re
        );
    }

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

    let data = format!(
        "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{:.6}\"\n",
        qs,
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
    fs::write(format!("iqp_{}_{}", qs, seed), data).expect("Unable to write file");

    Ok(())
}
