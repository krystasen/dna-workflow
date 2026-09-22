# VMD 1.9.3 text-mode clustering. Called by vmd_workflow.py only.
namespace eval ::vmd_cluster {}

proc ::vmd_cluster::write_frame_pdb {molid frame output_path} {
    set all_atoms [atomselect $molid all frame $frame]
    $all_atoms writepdb $output_path
    $all_atoms delete
}

proc ::vmd_cluster::pair_distance {molid selection distfunc frame_a frame_b} {
    set atoms_a [atomselect $molid $selection frame $frame_a]
    set atoms_b [atomselect $molid $selection frame $frame_b]
    if {[$atoms_a num] != [$atoms_b num]} {
        $atoms_a delete
        $atoms_b delete
        error "selection atom count changes between frames $frame_a and $frame_b"
    }
    switch -- $distfunc {
        rmsd {
            set distance [measure rmsd $atoms_a $atoms_b]
        }
        fitrmsd {
            set original_coordinates [$atoms_a get {x y z}]
            set transform [measure fit $atoms_a $atoms_b]
            $atoms_a move $transform
            set distance [measure rmsd $atoms_a $atoms_b]
            $atoms_a set {x y z} $original_coordinates
        }
        rgyrd {
            set distance [expr {abs([measure rgyr $atoms_a] - [measure rgyr $atoms_b])}]
        }
        default {
            $atoms_a delete
            $atoms_b delete
            error "unsupported VMD cluster distfunc: $distfunc"
        }
    }
    $atoms_a delete
    $atoms_b delete
    return $distance
}

proc ::vmd_cluster::medoid {molid selection distfunc members {distances {}}} {
    if {[llength $members] == 1} {
        return [lindex $members 0]
    }
    array set total_distance {}
    foreach frame $members {
        set total_distance($frame) 0.0
    }
    for {set i 0} {$i < [llength $members]} {incr i} {
        set frame_a [lindex $members $i]
        for {set j [expr {$i + 1}]} {$j < [llength $members]} {incr j} {
            set frame_b [lindex $members $j]
            if {[dict size $distances]} {
                set distance [dict get $distances [list $frame_a $frame_b]]
            } else {
                set distance [::vmd_cluster::pair_distance $molid $selection $distfunc $frame_a $frame_b]
            }
            set total_distance($frame_a) [expr {$total_distance($frame_a) + $distance}]
            set total_distance($frame_b) [expr {$total_distance($frame_b) + $distance}]
        }
    }
    set best_frame [lindex $members 0]
    set best_score $total_distance($best_frame)
    foreach frame [lrange $members 1 end] {
        if {$total_distance($frame) < $best_score || ($total_distance($frame) == $best_score && $frame < $best_frame)} {
            set best_frame $frame
            set best_score $total_distance($frame)
        }
    }
    return $best_frame
}

proc ::vmd_cluster::gromos_groups {distances count cutoff num} {
    # Seed-radius clustering, not a maximum pairwise cluster diameter.
    set remaining {}
    for {set i 0} {$i < $count} {incr i} {lappend remaining $i}
    set groups {}
    for {set c 0} {$c < $num && [llength $remaining]} {incr c} {
        set best {}
        foreach i $remaining {
            set neighbors {}
            foreach j $remaining {
                if {$i == $j || [dict get $distances [lsort -integer [list $i $j]]] <= $cutoff} {
                    lappend neighbors $j
                }
            }
            if {[llength $neighbors] > [llength $best]} {set best $neighbors}
        }
        lappend groups $best
        set next {}
        foreach i $remaining {
            if {[lsearch -integer -sorted $best $i] < 0} {lappend next $i}
        }
        set remaining $next
    }
    lappend groups $remaining
    return $groups
}

proc ::vmd_cluster::run {pdb xtc output selection distfunc cutoff num first last step {algorithm gromos}} {
    foreach path [list $pdb $xtc] {
        if {![file isfile $path]} {
            error "input does not exist: $path"
        }
    }
    if {![file isdirectory $output]} {
        error "output directory was not reserved: $output"
    }
    if {![string is double -strict $cutoff] || $cutoff <= 0} {
        error "cutoff must be a positive Angstrom value"
    }
    if {![string is integer -strict $num] || $num < 1} {
        error "num must be a positive integer"
    }
    if {![string is integer -strict $first] || $first < 0 || ![string is integer -strict $last] || ![string is integer -strict $step] || $step < 1} {
        error "first/last/step must be valid frame indices and a positive step"
    }
    if {[lsearch -exact {rmsd fitrmsd rgyrd} $distfunc] < 0} {
        error "unsupported VMD cluster distfunc: $distfunc"
    }
    if {[lsearch -exact {gromos vmd} $algorithm] < 0} {error "unknown clustering algorithm: $algorithm"}

    if {$last != -1 && $last < $first} {
        error "last must be -1 or no smaller than first"
    }
    set molid [mol new $pdb type pdb waitfor all]
    animate delete beg 0 end 0 $molid
    # Read only the requested frames. VMD cluster then sees a contiguous local frame range.
    mol addfile $xtc type xtc first $first last $last step $step waitfor all molid $molid
    set sampled_count [molinfo $molid get numframes]
    if {$sampled_count < 1} {
        error "dna_whole.xtc has no readable frame for first=$first last=$last step=$step"
    }
    if {$last != -1} {
        set requested_count [expr {($last - $first) / $step + 1}]
        if {$sampled_count != $requested_count} {
            error "requested XTC frames were not all readable: first=$first last=$last step=$step expected_count=$requested_count loaded_count=$sampled_count"
        }
    }
    set sampled_frames {}
    for {set local_frame 0} {$local_frame < $sampled_count} {incr local_frame} {
        lappend sampled_frames [expr {$first + $local_frame * $step}]
    }
    if {$num > $sampled_count} {
        error "num cannot exceed the selected sample-frame count ($sampled_count)"
    }

    set atoms [atomselect $molid $selection frame 0]
    if {[$atoms num] < 1} {
        $atoms delete
        error "selection contains no atoms in the first selected frame: $selection"
    }
    set selected_indices [$atoms get index]
    set fixed_selection "index [join $selected_indices { }]"
    set distances {}
    if {$algorithm eq "gromos"} {
        # ponytail: exact O(N^2) distances; increase sampling step for long trajectories.
        set matrix_file [open [file join $output pair_distances.csv] w]
        puts $matrix_file "frame_i,frame_j,distance_A"
        for {set i 0} {$i < $sampled_count} {incr i} {
            for {set j [expr {$i+1}]} {$j < $sampled_count} {incr j} {
                set d [::vmd_cluster::pair_distance $molid $fixed_selection $distfunc $i $j]
                dict set distances [list $i $j] $d
                puts $matrix_file "[lindex $sampled_frames $i],[lindex $sampled_frames $j],$d"
            }
            if {$i % 50 == 0} {puts "Pairwise distances: $i / $sampled_count"; flush stdout}
        }
        close $matrix_file
        set groups [::vmd_cluster::gromos_groups $distances $sampled_count $cutoff $num]
    } else {
        set groups [measure cluster $atoms cutoff $cutoff num $num distfunc $distfunc first 0 last -1 step 1]
    }
    $atoms delete
    if {[llength $groups] < 1} {
        error "VMD measure cluster returned no groups"
    }
    set unclustered [lindex $groups end]
    set raw_clusters [lrange $groups 0 end-1]
    set reported_frames $unclustered
    foreach members $raw_clusters {
        set reported_frames [concat $reported_frames $members]
    }
    set expected_frames {}
    for {set local_frame 0} {$local_frame < $sampled_count} {incr local_frame} {
        lappend expected_frames $local_frame
    }
    set observed_frames [lsort -integer $reported_frames]
    if {[llength $reported_frames] != [llength [lsort -integer -unique $reported_frames]] || $expected_frames ne $observed_frames} {
        error "measure cluster membership does not account for every loaded sample position exactly once; expected_positions=$expected_frames reported_positions=$observed_frames raw_groups=$groups"
    }

    set entries {}
    set raw_index 0
    foreach members $raw_clusters {
        incr raw_index
        if {[llength $members] > 0} {
            lappend entries [list [llength $members] $raw_index $members]
        }
    }
    set entries [lsort -integer -decreasing -index 0 $entries]
    set total_sampled $sampled_count
    set clusters_handle [open [file join $output clusters.csv] w]
    set members_handle [open [file join $output members.csv] w]
    puts $clusters_handle "cluster_id,representative_frame,time_ps,member_count,occupancy,representative_kind,status"
    puts $members_handle "cluster_id,frame,time_ps,status"
    set cluster_sizes {}
    set cluster_id 0
    foreach entry $entries {
        incr cluster_id
        set members [lindex $entry 2]
        set members [lsort -integer $members]
        set representative_position [::vmd_cluster::medoid $molid $fixed_selection $distfunc $members $distances]
        set representative [lindex $sampled_frames $representative_position]
        set size [llength $members]
        set occupancy [expr {double($size) / $total_sampled}]
        lappend cluster_sizes $size
        puts $clusters_handle "$cluster_id,$representative,null,$size,[format %.6f $occupancy],medoid,cluster"
        foreach position $members {
            puts $members_handle "$cluster_id,[lindex $sampled_frames $position],null,cluster"
        }
        set cluster_path [file join $output [format "cluster_%03d.pdb" $cluster_id]]
        ::vmd_cluster::write_frame_pdb $molid $representative_position $cluster_path
        if {$cluster_id == 1} {
            ::vmd_cluster::write_frame_pdb $molid $representative_position [file join $output dominant.pdb]
        }
    }
    set unclustered_count [llength $unclustered]
    set unclustered_occupancy [expr {double($unclustered_count) / $total_sampled}]
    puts $clusters_handle "unclustered,,null,$unclustered_count,[format %.6f $unclustered_occupancy],,unclustered"
    foreach position $unclustered {
        puts $members_handle "unclustered,[lindex $sampled_frames $position],null,unclustered"
    }
    close $clusters_handle
    close $members_handle

    set summary [open [file join $output summary.txt] w]
    puts $summary "status=success"
    puts $summary "algorithm=$algorithm"
    puts $summary "selected_atom_count=[llength $selected_indices]"
    puts $summary "gromos_cutoff_definition=distance to greedy seed; exported medoid is not necessarily seed"
    puts $summary "selection=$selection"
    puts $summary "distfunc=$distfunc"
    puts $summary "cutoff_A=$cutoff"
    puts $summary "requested_num=$num"
    puts $summary "first=$first"
    puts $summary "last=$last"
    puts $summary "step=$step"
    puts $summary "last_sampled_frame=[lindex $sampled_frames end]"
    puts $summary "xtc_frame_count=not read (filtered XTC load)"
    puts $summary "sampled_frame_count=$total_sampled"
    puts $summary "sampled_frame_indices=[join $sampled_frames { }]"
    puts $summary "measure_cluster_index_mode=loaded_sample_position_mapped_to_xtc_frame_index"
    puts $summary "cluster_count=[llength $entries]"
    puts $summary "cluster_member_counts=[join $cluster_sizes { }]"
    puts $summary "unclustered_count=$unclustered_count"
    puts $summary "accounted_frame_count=[expr {[llength $reported_frames]}]"
    puts $summary "time_ps=null"
    puts $summary "time_source=not available from VMD; see original XTC"
    puts $summary "representative=real frame medoid by exact all-member pairwise $distfunc mean minimization"
    if {$cluster_id > 0} {
        puts $summary "dominant_pdb=dominant.pdb (cluster_001 medoid)"
    } else {
        puts $summary "dominant_pdb=null (no classified cluster; unclustered frames are not a normal cluster)"
    }
    close $summary
    set marker [open [file join $output SUCCESS] w]
    puts $marker {action=cluster}
    close $marker
    mol delete $molid
}
