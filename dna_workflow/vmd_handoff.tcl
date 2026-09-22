# VMD Tk Console: cd {D:/path/to/job}; source vmd_handoff.tcl
# This loads the DNA-only trajectory. Clustering/representative selection remains an explicit user step.
set dna_mol [mol new dna.pdb type pdb waitfor all]
# Remove PDB's standalone coordinate frame: frame 0 must mean XTC frame 0.
animate delete beg 0 end 0 $dna_mol
mol addfile dna_whole.xtc type xtc waitfor all molid $dna_mol
mol delrep 0 $dna_mol
mol representation Licorice 0.2 12 12
mol selection {not hydrogen}
mol color ResName
mol addrep $dna_mol

proc export_dna_frame {molid frame output_path time_ps} {
    if {![string is integer -strict $frame] || $frame < 0 || $frame >= [molinfo $molid get numframes]} {
        error "frame must be a valid zero-based XTC frame index"
    }
    if {![string is double -strict $time_ps] || $time_ps < 0} {
        error "Record the actual XTC time in ps; do not assume frame number equals time"
    }
    if {[file exists $output_path] || [file exists "${output_path}.selection.txt"]} {
        error "Output exists; choose a new filename"
    }
    set selection [atomselect $molid all frame $frame]
    $selection writepdb $output_path
    $selection delete
    set f [open "${output_path}.selection.txt" w]
    puts $f "xtc_frame_0based=$frame\ntime_ps=$time_ps\nsource=dna_whole.xtc\nselection=all DNA atoms including hydrogens"
    close $f
    puts "Exported real frame $frame at $time_ps ps, with hydrogens preserved."
}
puts {Inspect PBC and choose an actual representative frame. Then: export_dna_frame $dna_mol FRAME {selected.pdb} TIME_PS}
