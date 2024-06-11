use std::collections::{HashSet, HashMap};

fn uncommon_elements(a: &HashSet<i32>, b: &HashSet<i32>) -> HashSet<i32> {
    a.union(b).cloned().collect::<HashSet<_>>()
        .difference(&a.intersection(b).cloned().collect::<HashSet<_>>())
        .cloned().collect()
}

fn group_sets_with_min_uncommon_elements(sets: Vec<HashSet<i32>>) -> (Vec<HashSet<i32>>, HashMap<Vec<usize>, HashSet<i32>>) {
    let n = sets.len();
    let mut uncommon_matrix = vec![vec![HashSet::new(); n]; n];
    
    // Step 1: Calculate uncommon elements for each pair of sets
    for i in 0..n {
        for j in (i + 1)..n {
            uncommon_matrix[i][j] = uncommon_elements(&sets[i], &sets[j]);
            uncommon_matrix[j][i] = uncommon_matrix[i][j].clone();
        }
    }
    
    // Step 2: Initialize each set as its own group
    let mut groups: Vec<HashSet<usize>> = (0..n).map(|i| {
        let mut set = HashSet::new();
        set.insert(i);
        set
    }).collect();
    
    // Initialize a hashmap to track uncommon elements between groups
    let mut group_uncommon : HashMap<Vec<usize>, HashSet<i32>> = HashMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            group_uncommon.insert(vec![i, j], uncommon_matrix[i][j].clone());
        }
    }
    
    // Step 3: Iteratively merge groups
    while groups.len() > 1 {
        let mut min_uncommon = usize::MAX;
        let mut merge_candidates = None;
        
        // Find the pair of groups with the minimum number of uncommon elements
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let mut uncommon: HashSet<i32> = HashSet::new();
                
                for &x in &groups[i] {
                    for &y in &groups[j] {
                        if let Some(value) = group_uncommon.get(&vec![x.min(y), x.max(y)]) {
                            uncommon.extend(value);
                        }
                    }
                }
                
                if uncommon.len() < min_uncommon {
                    min_uncommon = uncommon.len();
                    merge_candidates = Some((i, j));
                }
            }
        }
        
        let (i, j) = match merge_candidates {
            Some(pair) => pair,
            None => break,
        };
        
        // Merge groups[i] and groups[j]
        let merged_group = groups[i].union(&groups[j]).cloned().collect();
        groups[i] = merged_group;
        
        // Update the group_uncommon hashmap
        for k in 0..groups.len() {
            if k == i || k == j {
                continue;
            }
            
            let new_group = &groups[i];
            let existing_group = &groups[k];
            
            let mut new_uncommon = HashSet::new();
            for &x in new_group {
                for &y in existing_group {
                    new_uncommon.extend(uncommon_elements(&sets[x], &sets[y]));
                }
            }
            group_uncommon.insert(new_group.union(existing_group).cloned().collect(), new_uncommon);
        }
        
        groups.remove(j);
    }
    
    // Construct the final grouped sets
    let final_groups: Vec<HashSet<i32>> = groups.into_iter().map(|group| {
        group.into_iter().flat_map(|idx| sets[idx].clone()).collect()
    }).collect();
    
    // Return the final groups and the group_uncommon hashmap
    (final_groups, group_uncommon)
}

fn main() {
    // Example usage:
    let sets = vec![
        vec![1, 2, 3].into_iter().collect(),
        vec![2, 3, 4].into_iter().collect(),
        vec![4, 5, 6].into_iter().collect(),
        vec![2, 4, 7].into_iter().collect(),
    ];
    
    let (result, group_uncommon) = group_sets_with_min_uncommon_elements(sets);
    
    println!("Final Groups:");
    for group in result {
        println!("{:?}", group);
    }

    println!("Group Uncommon Elements:");
    for (v, uncommon) in group_uncommon {
        println!("Group ({:?}): {:?}", v, uncommon);
    }
}
