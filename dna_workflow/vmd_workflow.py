"""Run local VMD text sessions for DNA console scripts and RMSD clustering."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_VMD = Path(r"D:\software\vmd 1.9.3\vmd.exe")
SUCCESS_TEXT = "__VMD_WORKFLOW_SUCCESS__"
DEFAULT_TIMEOUT = 900


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def tcl_word(value: str) -> str:
    """Emit one literal Tcl word without evaluating a path or selection as Tcl."""
    if "\n" in value or "\r" in value:
        raise ValueError("paths and selections must not contain newlines")
    if not value:
        return "{}"
    escaped = {
        "\\": "\\\\", " ": "\\ ", "\t": "\\t", "$": "\\$", "[": "\\[",
        "]": "\\]", ";": "\\;", "\"": "\\\"", "{": "\\{", "}": "\\}", "#": "\\#",
    }
    return "".join(escaped.get(char, char) for char in value)


def _assignment(name: str, value: str) -> str:
    return f"set {name} {tcl_word(value)}"


def _prepare_job(job_value: str) -> tuple[Path, Path, Path]:
    job = Path(job_value).expanduser().resolve()
    pdb = job / "dna.pdb"
    xtc = job / "dna_whole.xtc"
    missing = [str(path) for path in (pdb, xtc) if not path.is_file()]
    if missing:
        raise ValueError("job requires dna.pdb and dna_whole.xtc; missing: " + ", ".join(missing))
    return job, pdb, xtc


def _reserve_output(output_value: str) -> Path:
    output = Path(output_value).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output must be a new directory: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output.parent}")
    output.mkdir()
    return output


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _cluster_script(job: Path, pdb: Path, xtc: Path, output: Path, args: argparse.Namespace) -> str:
    return "\n".join((
        _assignment("workflow_cluster_tcl", str(HERE / "vmd_cluster.tcl")),
        _assignment("workflow_pdb", str(pdb)),
        _assignment("workflow_xtc", str(xtc)),
        _assignment("workflow_output", str(output)),
        _assignment("workflow_selection", args.selection),
        _assignment("workflow_distfunc", args.distfunc),
        _assignment("workflow_cutoff", str(args.cutoff)),
        _assignment("workflow_num", str(args.num)),
        _assignment("workflow_first", str(args.first)),
        _assignment("workflow_last", str(args.last)),
        _assignment("workflow_step", str(args.step)),
        _assignment("workflow_algorithm", args.algorithm),
        "proc workflow_fail {message} {",
        "    puts stderr \"__VMD_WORKFLOW_ERROR__ $message\"",
        "    exit 1",
        "}",
        "if {[catch {",
        "    source $workflow_cluster_tcl",
        "    ::vmd_cluster::run $workflow_pdb $workflow_xtc $workflow_output $workflow_selection $workflow_distfunc $workflow_cutoff $workflow_num $workflow_first $workflow_last $workflow_step $workflow_algorithm",
        "} workflow_error]} {",
        "    workflow_fail $workflow_error",
        "}",
        f"puts {SUCCESS_TEXT}",
        "exit",
        "",
    ))


def _console_script(pdb: Path, xtc: Path, output: Path, script_file: Path) -> str:
    return "\n".join((
        "namespace eval ::vmd_workflow { variable job; variable output; variable dna_mol -1 }",
        _assignment("::vmd_workflow::job", str(pdb.parent)),
        _assignment("::vmd_workflow::output", str(output)),
        _assignment("workflow_pdb", str(pdb)),
        _assignment("workflow_xtc", str(xtc)),
        _assignment("workflow_user_script", str(script_file)),
        "proc ::vmd_workflow::write_success {} {",
        "    set handle [open [file join $::vmd_workflow::output SUCCESS] w]",
        "    puts $handle {action=console}",
        "    close $handle",
        "}",
        "proc workflow_fail {message} {",
        "    puts stderr \"__VMD_WORKFLOW_ERROR__ $message\"",
        "    exit 1",
        "}",
        "if {[catch {",
        "    set ::vmd_workflow::dna_mol [mol new $workflow_pdb type pdb waitfor all]",
        "    animate delete beg 0 end 0 $::vmd_workflow::dna_mol",
        "    mol addfile $workflow_xtc type xtc waitfor all molid $::vmd_workflow::dna_mol",
        "    source $workflow_user_script",
        "    ::vmd_workflow::write_success",
        "} workflow_error]} {",
        "    workflow_fail $workflow_error",
        "}",
        f"puts {SUCCESS_TEXT}",
        "exit",
        "",
    ))


def _write_logs(output: Path, argv: list[str], stdout: str, stderr: str, returncode: int | None) -> None:
    _write(output / "vmd.stdout.log", stdout)
    _write(output / "vmd.stderr.log", stderr)
    _write(output / "workflow.log", "command=" + subprocess.list2cmdline(argv) + "\nreturncode=" + str(returncode) + "\n")


def _run_vmd(vmd: Path, script: Path, job: Path, output: Path, timeout: int) -> None:
    startup = output / "vmd_startup.tcl"
    _write(startup, "")
    argv = [str(vmd), "-startup", str(startup), "-dispdev", "text", "-e", str(script)]
    try:
        result = subprocess.run(argv, cwd=job, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        _write_logs(output, argv, stdout, stderr, None)
        raise RuntimeError(f"VMD timed out after {timeout} seconds; the text-session process was terminated.") from None
    _write_logs(output, argv, result.stdout, result.stderr, result.returncode)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"VMD exited with code {result.returncode}; see {output / 'vmd.stderr.log'}")
    if SUCCESS_TEXT not in result.stdout or not (output / "SUCCESS").is_file():
        raise RuntimeError(f"VMD returned without the required success marker; see {output / 'workflow.log'}")


def _vmd_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"VMD executable was not found: {path}")
    return path


def run_cluster(args: argparse.Namespace) -> None:
    job, pdb, xtc = _prepare_job(args.job)
    output = _reserve_output(args.output)
    vmd = _vmd_path(args.vmd)
    script = output / "run_cluster.tcl"
    _write(script, _cluster_script(job, pdb, xtc, output, args))
    _run_vmd(vmd, script, job, output, args.timeout)
    required = [output / name for name in ("summary.txt", "clusters.csv", "members.csv")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("VMD reported success but cluster outputs are missing: " + ", ".join(missing))
    print(f"Cluster outputs: {output}")


def run_console(args: argparse.Namespace) -> None:
    job, pdb, xtc = _prepare_job(args.job)
    script_file = Path(args.script_file).expanduser().resolve()
    if not script_file.is_file():
        raise ValueError(f"Tcl script was not found: {script_file}")
    output = _reserve_output(args.output)
    vmd = _vmd_path(args.vmd)
    submitted = output / "submitted.tcl"
    shutil.copyfile(script_file, submitted)
    _write(output / "console_context.txt", "session=independent VMD text session\n"
           "note=This is Tcl execution in VMD, not a Tk window embedded in the web GUI.\n"
           f"job={job}\nscript_file={script_file}\nsubmitted_script={submitted}\n")
    script = output / "run_console.tcl"
    _write(script, _console_script(pdb, xtc, output, submitted))
    _run_vmd(vmd, script, job, output, args.timeout)
    print(f"Console outputs: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    cluster = commands.add_parser("cluster", help="run VMD measure cluster and export real medoid frames")
    cluster.add_argument("--job", required=True, help="directory containing dna.pdb and dna_whole.xtc")
    cluster.add_argument("--output", required=True, help="new output directory")
    cluster.add_argument("--selection", default="not hydrogen")
    cluster.add_argument("--algorithm", choices=("gromos", "vmd"), default="gromos",
                         help="independent greedy seed-radius clustering, or legacy VMD measure cluster")
    cluster.add_argument("--distfunc", choices=("rmsd", "fitrmsd", "rgyrd"), default="fitrmsd")
    cluster.add_argument("--cutoff", type=positive_float, default=2.5, help="cluster cutoff in Angstrom")
    cluster.add_argument("--num", type=positive_int, default=5)
    cluster.add_argument("--first", type=nonnegative_int, default=0)
    cluster.add_argument("--last", type=int, default=-1, help="last frame, or -1 for the last XTC frame")
    cluster.add_argument("--step", type=positive_int, default=10)
    cluster.add_argument("--vmd", default=str(DEFAULT_VMD))
    cluster.add_argument("--timeout", type=positive_int, default=DEFAULT_TIMEOUT)
    cluster.set_defaults(func=run_cluster)

    console = commands.add_parser("console", help="run an explicitly selected local Tcl script in a fresh VMD text session")
    console.add_argument("--job", required=True, help="directory containing dna.pdb and dna_whole.xtc")
    console.add_argument("--output", required=True, help="new output directory")
    console.add_argument("--script-file", required=True, help="local Tcl file chosen explicitly by the GUI user")
    console.add_argument("--vmd", default=str(DEFAULT_VMD))
    console.add_argument("--timeout", type=positive_int, default=DEFAULT_TIMEOUT)
    console.set_defaults(func=run_console)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
