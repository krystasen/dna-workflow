"""Run inside an allocated server job. No scheduler assumptions and no automatic submission.

Use prepare_em/nvt/npt/md locally, then verify_em/nvt/npt/md after server returns.
Only explicit legacy em/nvt/npt/md CLI commands run mdrun; the GUI never exposes them.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import errno
import json
import math
import os
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
import time

if os.name == "posix":
    import pty

from dna_workflow import read_pdb, restore_identity, save_json, validate_dna, pdb_text


MD_STAGES = ("em", "nvt", "npt", "md")
STAGES = ["topology", "solvate", *MD_STAGES,
          *(f"prepare_{s}" for s in MD_STAGES), *(f"verify_{s}" for s in MD_STAGES), "handoff"]


def return_files(stage):
    return [f"{stage}.{ext}" for ext in (["tpr", "gro", "log"] if stage == "em" else ["tpr", "gro", "log", "cpt"])]


def file_state(job, names):
    # Track real file replacements without hashing large TPR/CPT files on every GUI poll.
    return {name: [s.st_size, s.st_mtime_ns // 1_000_000] for name in names if (s := (job / name).stat())}


def verified_return(job, stage):
    marker = job / f"{stage}.verified.json"
    try:
        report = json.loads(marker.read_text(encoding="utf-8"))
        names = [*return_files(stage), f"{stage}.mdp"]
        return report.get("files") == file_state(job, names)
    except (OSError, ValueError):
        return False


def mdp_settings(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(";", 1)[0]
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower().replace("_", "-")] = value.strip()
    return values


def gro_atoms(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[1])
    if len(lines) != count + 3:
        raise ValueError(f"{path.name}: incomplete GRO file")
    names = []
    for line in lines[2:2+count]:
        names.append((line[5:10].strip(), line[10:15].strip()))
        if not all(math.isfinite(float(line[i:i+8])) for i in (20, 28, 36)):
            raise ValueError(f"{path.name}: non-finite coordinates")
    return names


def verify_return(job, stage, command):
    """Verify imported server outputs; never run MD and never depend on a local completion marker."""
    names = [*return_files(stage), f"{stage}.mdp"]
    missing = [n for n in names if not (job / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{stage.upper()} 回传缺少: {', '.join(missing)}")
    settings = mdp_settings(job / f"{stage}.mdp")
    saved = job / f"{stage}.submitted.mdp"
    if saved.is_file() and mdp_settings(saved) != settings:
        raise ValueError(f"{stage}.mdp changed after TPR preparation; use a separate job for changed parameters.")
    tpr = command(["dump", "-s", f"{stage}.tpr"], f"{stage}-verify-tpr", capture=True, echo=False)
    fields = dict(re.findall(r"^\s*([\w-]+)\s*=\s*([^\r\n]+)", tpr, re.M))
    steps = int(fields["nsteps"])
    if steps != int(settings["nsteps"]):
        raise ValueError(f"{stage}.tpr nsteps={steps} differs from {stage}.mdp")
    for key in ("integrator", "tcoupl", "pcoupl"):
        if key in settings and fields.get(key, "").strip().lower() != settings[key].lower():
            raise ValueError(f"{stage}.tpr {key} differs from {stage}.mdp; wrong stage or input returned.")
    actual_atoms = gro_atoms(job / f"{stage}.gro")
    if len(actual_atoms) != int(fields["natoms"]):
        raise ValueError(f"{stage}.gro atom count differs from TPR")
    base = job / "solv_ions.gro"
    if base.is_file() and actual_atoms != gro_atoms(base):
        raise ValueError(f"{stage}.gro atom order/composition differs from this job")
    log = (job / f"{stage}.log").read_text(encoding="utf-8", errors="replace")
    report = {"stage": stage, "source": names, "equilibration_proven": False,
              "validation": "Imported output verification only; no mdrun executed"}
    if stage == "em":
        forces = re.findall(r"Maximum force\s*=\s*([0-9.eE+-]+)", log)
        if not forces or not math.isfinite(float(forces[-1])) or float(forces[-1]) > float(settings.get("emtol", 1000)):
            raise ValueError("EM Fmax did not reach the configured emtol; cannot start NVT")
        report["Fmax_kJ_mol_nm"] = float(forces[-1])
    else:
        nsteps, dt = tpr_settings(tpr)
        if not math.isclose(dt, float(settings["dt"]), rel_tol=1e-9):
            raise ValueError(f"{stage}.tpr dt differs from MDP")
        if "Finished mdrun" not in log:
            raise ValueError(f"{stage}.log lacks Finished mdrun; return completed output or resume on the server")
        checkpoint = command(["dump", "-cp", f"{stage}.cpt"], f"{stage}-verify-cpt", capture=True, echo=False)
        step = completed_step(checkpoint)
        target = int(fields.get("init-step", 0)) + nsteps
        cpt_atoms = re.search(r"^#atoms\s*=\s*(\d+)", checkpoint, re.M)
        if not cpt_atoms or int(cpt_atoms[1]) != len(actual_atoms):
            raise ValueError(f"{stage}.cpt atom count differs from returned coordinates")
        cpt_time = re.search(r"^t\s*=\s*([\d.eE+-]+)", checkpoint, re.M)
        if step < target or not cpt_time or not math.isclose(float(cpt_time[1]), float(fields.get("tinit", 0)) + step * dt, abs_tol=1e-5):
            raise ValueError(f"{stage.upper()} checkpoint incomplete or inconsistent: step {step}/{target}; resume on the server")
        report.update(step=step, target_steps=target, dt_ps=dt, duration_ps=nsteps * dt)
    report["files"] = file_state(job, names)
    save_json(job / f"{stage}.verified.json", report)
    print(f"{stage.upper()} 回传核验通过；未运行本机模拟。", flush=True)
    return report


def completed_step(text: str) -> int:
    match = re.search(r"^\s*step\s*=\s*(\d+)\s*$", text, flags=re.M)
    if not match:
        raise RuntimeError("Cannot read checkpoint step; inspect checkpoint.dump.log.")
    return int(match[1])


def restore_trajectory_identity(exported, topology, strands):
    """trjconv can drop chain IDs; restore identity by the unchanged topology atom order."""
    reference = restore_identity(topology, strands)
    if [(a.name, a.residue) for a in exported] != [(a.name, a.residue) for a in reference]:
        raise ValueError("Trajectory atoms differ from topology; do not guess chain membership.")
    return [replace(a, xyz=b.xyz) for a, b in zip(reference, exported)]


def stream_process(argv, cwd: Path, log_path: Path, input_text: str | None = None, capture=False, echo=True) -> str:
    """Append live child output to both this process stdout and the stage log."""
    if echo:
        print(shlex.join(argv), flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(argv) + "\n")
        stream.flush()
        proc = subprocess.Popen(argv, cwd=cwd, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if input_text is not None:
            assert proc.stdin is not None
            proc.stdin.write(input_text)
            proc.stdin.close()
        assert proc.stdout is not None
        output = []
        for line in proc.stdout:
            stream.write(line)
            if echo:
                stream.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            if capture:
                output.append(line)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return "".join(output)


def choose_terminals(argv, cwd: Path, log_path: Path, selection_log: Path, strands: int, timeout: float) -> None:
    """Select the actual 5TER/3TER menu entries through a WSL pseudo-terminal."""
    if os.name != "posix":
        raise RuntimeError("--noninteractive-terminals requires a POSIX/WSL Python runtime.")
    expected = [label for _ in range(strands) for label in ("5TER", "3TER")]
    report = {"expected": expected, "selections": [], "timeout_s": timeout, "validation": "menu labels selected dynamically"}
    print(shlex.join(argv), flush=True)
    master, slave = pty.openpty()
    proc = None
    try:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(shlex.join(argv) + "\n")
            stream.flush()
            proc = subprocess.Popen(argv, cwd=cwd, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
            os.close(slave)
            slave = None
            pending = ""
            last_output = time.monotonic()
            while proc.poll() is None:
                ready, _, _ = select.select([master], [], [], 0.2)
                if not ready:
                    if time.monotonic() - last_output > timeout:
                        proc.terminate()
                        raise TimeoutError(f"pdb2gmx terminal prompt/output stalled for {timeout:g} s.")
                    continue
                try:
                    raw = os.read(master, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:  # Linux PTY reports EIO after the child closes its slave side.
                        break
                    raise
                chunk = raw.decode("utf-8", errors="replace")
                if not chunk:
                    continue
                last_output = time.monotonic()
                stream.write(chunk)
                stream.flush()
                sys.stdout.write(chunk)
                sys.stdout.flush()
                pending += chunk
                while True:
                    # GROMACS 2025.4 choose_ter prints no textual input prompt: its menu is
                    # `Select start/end terminus type ...` followed by numbered choices.
                    prompt = re.search(r"Select\s+(start|end)\s+terminus\s+type\s+for\s+[^\r\n]+", pending, flags=re.I)
                    if not prompt:
                        break
                    menu_end = re.search(r"^\s*\d+\s*:\s*None\s*\r?$", pending[prompt.end():], flags=re.I | re.M)
                    if not menu_end:
                        break  # The prompt arrived, but wait until its complete current menu is printed.
                    block_end = prompt.end() + menu_end.end()
                    block, pending = pending[:block_end], pending[block_end:]
                    if len(report["selections"]) >= len(expected):
                        raise RuntimeError("pdb2gmx asked for more terminal menus than the strand manifest; refusing to guess an answer.")
                    wanted = "5TER" if prompt.group(1).lower() == "start" else "3TER"
                    choices = re.findall(r"^\s*(\d+)\s*:\s*(.+?)\s*$", block[prompt.end():], flags=re.M)
                    number = next((value for value, label in choices
                                   if re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", label, flags=re.I)), None)
                    if number is None:
                        raise RuntimeError(f"pdb2gmx menu does not offer required {wanted}; refusing to substitute another terminus.")
                    report["selections"].append({"prompt": prompt.group(0), "wanted": wanted,
                                                 "selected_number": int(number), "choices": choices})
                    os.write(master, (number + "\n").encode("ascii"))
            while True:
                ready, _, _ = select.select([master], [], [], 0)
                if not ready:
                    break
                try:
                    raw = os.read(master, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                chunk = raw.decode("utf-8", errors="replace")
                if not chunk:
                    break
                stream.write(chunk)
                stream.flush()
                sys.stdout.write(chunk)
                sys.stdout.flush()
            code = proc.wait()
            if code:
                raise subprocess.CalledProcessError(code, argv)
            if len(report["selections"]) != len(expected):
                raise RuntimeError(f"pdb2gmx ended after {len(report['selections'])}/{len(expected)} terminal selections.")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if slave is not None:
            os.close(slave)
        try:
            os.close(master)
        except OSError:
            pass
        report["exit_code"] = proc.returncode if proc is not None else None
        save_json(selection_log, report)


def tpr_settings(text: str) -> tuple[int, float]:
    nsteps = re.search(r"^\s*nsteps\s*=\s*(\d+)\s*$", text, flags=re.M)
    dt = re.search(r"^\s*dt\s*=\s*([0-9.eE+-]+)\s*$", text, flags=re.M)
    if not nsteps or not dt:
        raise RuntimeError("Cannot read nsteps/dt from md.tpr dump; inspect logs/md-tpr.log.")
    return int(nsteps.group(1)), float(dt.group(1))


def run(args):
    job = args.job.resolve()
    cfg = json.loads((job / "workflow.json").read_text(encoding="utf-8"))
    (job / "logs").mkdir(exist_ok=True)
    gmx = args.gmx

    def command(parts, label, input_text=None, capture=False, echo=True):
        argv = [gmx, *map(str, parts)]
        with (job / "logs" / "commands.log").open("a", encoding="utf-8") as commands:
            commands.write(shlex.join(argv) + "\n")
        return stream_process(argv, job, job / "logs" / f"{label}.log", input_text, capture, echo)

    def require(*names):
        for name in names:
            if not (job / name).is_file():
                raise FileNotFoundError(f"Missing {name}; finish the previous stage first.")

    def fresh(*names):
        for name in names:
            if (job / name).exists():
                raise FileExistsError(f"Refusing to overwrite {name}. Keep this run and build a new job if inputs changed.")

    def mark(stage, report):
        save_json(job / f"{stage}.complete.json", report)

    command(["--version"], "gromacs-version")
    stages = ["em", "nvt", "npt", "md"] if args.stage == "production" else [args.stage]
    for stage in stages:
        if stage == "topology":
            fresh("topol.top", "topology.pdb")
            print("Choose 5TER at each 5' end and 3TER at each 3' end (not None/5PHO).", flush=True)
            parts = ["pdb2gmx", "-f", "dna_input.pdb", "-o", "topology.pdb", "-p", "topol.top",
                     "-i", "posre.itp", "-ff", cfg["forcefield"].removesuffix(".ff"), "-water", "tip3p",
                     "-ignh", "-ter", "-chainsep", "id_or_ter", "-merge", "no"]
            if args.noninteractive_terminals:
                choose_terminals([gmx, *map(str, parts)], job, job / "logs" / "topology.log",
                                 job / "topology.terminal_selections.json", len(cfg["strands_5to3"]), args.terminal_timeout)
            else:
                command(parts, "topology")
            atoms = restore_identity(read_pdb(job / "topology.pdb"), cfg["strands_5to3"])
            check = validate_dna(atoms, cfg["strands_5to3"], job / cfg["forcefield"], hydrogens=True)
            (job / "dna_reference.pdb").write_text(pdb_text(atoms), encoding="ascii")
            check["validation"] = "pdb2gmx executed; all-atom CHARMM names, sequence, termini and bond lengths passed"
            mark(stage, check)
        elif stage == "solvate":
            require("topology.complete.json", "topology.pdb", "topol.top")
            fresh("boxed.gro", "solvated.gro", "ions.tpr", "solv_ions.gro")
            command(["editconf", "-f", "topology.pdb", "-o", "boxed.gro", "-c", "-d", cfg["padding_nm"],
                     "-bt", cfg.get("box_type", "dodecahedron")], "box")
            command(["solvate", "-cp", "boxed.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"], "solvate")
            # This TPR is ONLY for replacing water by ions; no dynamics or PME on a charged box.
            (job / "ions.mdp").write_text("integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n"
                                          "coulombtype = Cut-off\nrcoulomb = 1.2\nrvdw = 1.2\npbc = xyz\n", encoding="ascii")
            command(["grompp", "-f", "ions.mdp", "-c", "solvated.gro", "-p", "topol.top", "-o", "ions.tpr", "-po", "ions.processed.mdp"], "ions-grompp")
            command(["genion", "-s", "ions.tpr", "-o", "solv_ions.gro", "-p", "topol.top",
                     "-pname", "SOD", "-nname", "CLA", "-neutral", "-conc", cfg["salt_M"], "-seed", cfg["seed"]], "genion", "SOL\n")
            mark(stage, {"water": "CHARMM TIP3P", "additional_salt_M": cfg["salt_M"], "neutralize": True})
        elif stage in {*MD_STAGES, *(f"prepare_{s}" for s in MD_STAGES)}:
            preparing = stage.startswith("prepare_")
            md_stage = stage.removeprefix("prepare_")
            predecessor = {"em": "solvate", "nvt": "em", "npt": "nvt", "md": "npt"}[md_stage]
            if preparing and predecessor != "solvate":
                if not verified_return(job, predecessor):
                    verify_return(job, predecessor, command)
            else:
                require(predecessor + ".complete.json")
            coord = {"em": "solv_ions.gro", "nvt": "em.gro", "npt": "nvt.gro", "md": "npt.gro"}[md_stage]
            require(coord, "topol.top", md_stage + ".mdp")
            mdp = (job / f"{md_stage}.mdp").read_text()
            nsteps = int(re.search(r"^nsteps\s*=\s*(\d+)", mdp, re.M)[1])
            saved_mdp = job / f"{md_stage}.submitted.mdp"
            if saved_mdp.exists() and saved_mdp.read_text() != mdp:
                raise RuntimeError("MDP changed after TPR generation; create a new run rather than mixing restart parameters.")
            if (job / f"{md_stage}.tpr").exists() and not saved_mdp.exists():
                raise RuntimeError("Untracked TPR exists; inspect it rather than adopting unknown simulation parameters.")
            if not (job / f"{md_stage}.tpr").exists():
                parts = ["grompp", "-f", f"{md_stage}.mdp", "-c", coord, "-p", "topol.top", "-o", f"{md_stage}.tpr", "-po", f"{md_stage}.processed.mdp"]
                if md_stage in {"nvt", "npt"}:
                    parts += ["-r", "em.gro"]
                if md_stage in {"npt", "md"}:
                    require(predecessor + ".cpt")
                    parts += ["-t", predecessor + ".cpt"]
                command(parts, md_stage + "-grompp")
                saved_mdp.write_text(mdp, encoding="ascii")
            if preparing:
                print(f"{md_stage}.tpr 已准备（只执行grompp）。请在超算运行: gmx mdrun -deffnm {md_stage}", flush=True)
                print("回传到本任务目录后核验: " + ", ".join(return_files(md_stage)), flush=True)
                continue
            if (job / f"{md_stage}.complete.json").exists():
                require(md_stage + ".gro", md_stage + ".log")
                print(f"{md_stage}: already validated complete; skipped", flush=True)
                continue
            parts = ["mdrun", "-deffnm", md_stage]
            if args.ntmpi:
                if Path(gmx).name == "gmx_mpi":
                    raise RuntimeError("--ntmpi is for thread-MPI gmx; do not set it with gmx_mpi.")
                parts += ["-ntmpi", args.ntmpi]
            if args.ntomp:
                parts += ["-ntomp", args.ntomp]
            if args.maxh:
                parts += ["-maxh", args.maxh]
            if (job / f"{md_stage}.cpt").exists():
                parts += ["-cpi", f"{md_stage}.cpt", "-append"]
            elif (job / f"{md_stage}.log").exists():
                raise RuntimeError(f"{md_stage}.log exists without checkpoint: inspect failure before retrying.")
            command(parts, md_stage + "-mdrun")
            require(md_stage + ".gro", md_stage + ".log")
            if md_stage == "em":
                log = (job / "em.log").read_text(errors="replace")
                forces = re.findall(r"Maximum force\s*=\s*([0-9.eE+-]+)", log)
                if not forces or float(forces[-1]) > 1000:
                    raise RuntimeError("EM did not reach Fmax <=1000 kJ/mol/nm; inspect em.log before NVT.")
                report = {"Fmax_kJ_mol_nm": float(forces[-1])}
            else:
                require(md_stage + ".cpt")
                dump = command(["dump", "-cp", f"{md_stage}.cpt"], md_stage + "-checkpoint", capture=True, echo=False)
                step = completed_step(dump)
                print(f"{md_stage} checkpoint: step {step}/{nsteps}", flush=True)
                if step < nsteps:
                    raise RuntimeError(f"{md_stage} paused at step {step}/{nsteps}; rerun same stage to resume. No next stage started.")
                report = {"step": step, "target_steps": nsteps, "equilibration_proven": False}
            mark(md_stage, report)
        elif stage.startswith("verify_"):
            verify_return(job, stage.removeprefix("verify_"), command)
        elif stage == "handoff":
            # Imported server outputs may be valid trajectory inputs without a local completion marker.
            # `verify_md` is the separate, explicit proof-of-completion action.
            require("workflow.json", "topology.pdb", "md.tpr", "md.xtc")
            fresh("dna.ndx", "dna.tpr", "dna_molecules.xtc", "dna_clustered.xtc", "dna_whole.xtc", "dna.pdb", "handoff.complete.json")
            topology = restore_identity(read_pdb(job / "topology.pdb"), cfg["strands_5to3"])
            validate_dna(topology, cfg["strands_5to3"], job / cfg["forcefield"], hydrogens=True)
            count = len(topology)
            ids = list(range(1, count + 1))
            (job / "dna.ndx").write_text("[ DNA ]\n" + "\n".join(" ".join(map(str, ids[i:i+15])) for i in range(0, count, 15)) + "\n", encoding="ascii")
            # Cluster requires already-whole molecules; do not apply it directly to wrapped strands.
            command(["trjconv", "-s", "md.tpr", "-f", "md.xtc", "-n", "dna.ndx", "-o", "dna_molecules.xtc", "-pbc", "whole"], "dna-whole-molecules", "0\n")
            command(["convert-tpr", "-s", "md.tpr", "-n", "dna.ndx", "-o", "dna.tpr"], "dna-tpr", "0\n")
            clustered = "dna_molecules.xtc"
            if len(cfg["strands_5to3"]) > 1:
                clustered = "dna_clustered.xtc"
                command(["trjconv", "-s", "dna.tpr", "-f", "dna_molecules.xtc", "-n", "dna.ndx", "-o", clustered, "-pbc", "cluster"], "dna-pbc", "0\n0\n")
            # Translate the whole DNA group together; no per-molecule wrapping after clustering.
            command(["trjconv", "-s", "dna.tpr", "-f", clustered, "-n", "dna.ndx", "-o", "dna_whole.xtc",
                     "-center", "-boxcenter", "tric", "-pbc", "none"], "dna-center", "0\n0\n")
            command(["trjconv", "-s", "dna.tpr", "-f", "dna_whole.xtc", "-n", "dna.ndx", "-o", "dna.pdb", "-dump", "0"], "dna-reference", "0\n")
            reference = restore_trajectory_identity(read_pdb(job / "dna.pdb"), read_pdb(job / "topology.pdb"), cfg["strands_5to3"])
            validate_dna(reference, cfg["strands_5to3"], job / cfg["forcefield"], hydrogens=True)
            (job / "dna.pdb").write_text(pdb_text(reference), encoding="ascii")
            mark(stage, {"return_to_local": ["dna_whole.xtc", "dna.pdb", "dna.tpr", "topology.pdb", "workflow.json", "vmd_handoff.tcl", cfg["forcefield"]],
                         "centered": True, "box_center": "tric", "centering_method": "GROMACS selection bounding-box center",
                         "selection": "all DNA atoms including H; all strands together",
                         "atom_count": count, "strand_count": len(cfg["strands_5to3"]),
                         "processing": ["whole", *( ["cluster"] if len(cfg["strands_5to3"]) > 1 else []), "center without rewrapping"],
                         "production_md": "unverified unless md.complete.json or md.verified.json exists",
                         "note": "Inspect PBC reconstruction in VMD; post-processing success does not prove production MD duration or completion."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES + ["production"])
    parser.add_argument("--job", type=Path, default=Path.cwd())
    parser.add_argument("--gmx", default="gmx", help="Executable path, e.g. gmx or gmx_mpi; not a shell expression")
    parser.add_argument("--ntmpi", type=int, help="Thread-MPI ranks; use 1 with this local gmx to make ntomp the total CPU-thread count")
    parser.add_argument("--ntomp", type=int)
    parser.add_argument("--maxh", type=float, help="Stop cleanly within allocated wall time; resume same stage later")
    parser.add_argument("--noninteractive-terminals", action="store_true", help="Dynamically select actual 5TER/3TER menu labels")
    parser.add_argument("--terminal-timeout", type=float, default=60.0, help="Fail rather than wait indefinitely for pdb2gmx input")
    args = parser.parse_args()
    if args.ntomp is not None and args.ntomp < 1:
        parser.error("ntomp must be positive")
    if args.ntmpi is not None and args.ntmpi < 1:
        parser.error("ntmpi must be positive")
    if args.maxh is not None and args.maxh <= 0:
        parser.error("maxh must be positive")
    if args.terminal_timeout <= 0:
        parser.error("terminal-timeout must be positive")
    run(args)
