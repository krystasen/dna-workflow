"""Local-only browser GUI for the DNA workflow. No remote APIs or shell execution."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
import socket
from pathlib import Path, PurePosixPath
import subprocess
import sys
import threading
from urllib.parse import unquote, urlparse
import uuid
import webbrowser
from md_pipeline import MD_STAGES, return_files, verified_return
from dna_workflow import MD_MODELS, AMBER_ENV


HERE = Path(__file__).resolve().parent
GUI = HERE / "gui"
RUNTIME = Path(r"D:\data\DNA DoS\workflow_runtime\Scripts\python.exe")
DOS_RUNTIME = Path(r"D:\data\DNA DoS\dos_runtime\Scripts\python.exe")
DOS_SCRIPT = HERE.parent / "dna_dos_v2.py"
VMD = Path(r"D:\software\vmd 1.9.3\vmd.exe")
MAX_LOG = 200_000
DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8765,
    "python": str(RUNTIME),
    "dos_python": str(DOS_RUNTIME),
    "distribution": "Ubuntu-24.04",
    "gmx": "/opt/gromacs/2025.4/bin/gmx",
    "wsl_python": "/usr/bin/python3",
    "vmd": str(VMD),
    "ntmpi": 1,
    "ntomp": 2,
    "threading_note": "Local thread-MPI gmx uses -ntmpi 1; ntomp is the total CPU-thread count. gmx_mpi receives no -ntmpi.",
    "temperature": 310.0,
    "salt": 0.15,
    "nvt_ps": 100.0,
    "npt_ps": 1000.0,
    "md_ns": 100.0,
    "xtc_ps": 10.0,
    "padding": 1.5,
    "seed": 20260905,
    "md_model": "charmm36_tip3p",
    "amber_tleap": AMBER_ENV + "/bin/tleap",
    "amber_python": AMBER_ENV + "/bin/python",
    "salt_cation": "Na+",
}
PREPARATION_ACTIONS = {f"prepare_{s}" for s in MD_STAGES}
VERIFICATION_ACTIONS = {f"verify_{s}" for s in MD_STAGES}
ALLOWED_ACTIONS = {"topology", "solvate", "handoff", "gaussian", "open_folder", "open_vmd"} | PREPARATION_ACTIONS | VERIFICATION_ACTIONS
ANALYSIS_ACTIONS = {"cluster", "open_tk", "prepare_relax", "verify_relax"}
ALLOWED_ACTIONS |= ANALYSIS_ACTIONS
DOS_ACTIONS = {"dos_inspect", "dos_export", "dos_verify", "dos_plot", "dos_open"}
ALLOWED_ACTIONS |= DOS_ACTIONS
DOS_PARAMETERS = ("energy_min_eV", "energy_max_eV", "energy_step_eV", "eta_eV", "dos_bin_width_eV",
                  "gamma_contact_eV", "gamma_b_eV", "lambda_eV", "temperature_K", "spurious_min_eV", "spurious_max_eV")


class APIError(ValueError):
    def __init__(self, message: str, status=400):
        super().__init__(message)
        self.status = status


def required(data: dict, name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise APIError(f"Missing required field: {name}.")
    return value.strip()


def number(data: dict, name: str) -> float:
    try:
        value = float(data[name])
    except (KeyError, TypeError, ValueError):
        raise APIError(f"{name} must be a number.") from None
    if not math.isfinite(value):
        raise APIError(f"{name} must be finite.")
    return value


def reject_o6meg(value: str, field: str) -> None:
    compact = "".join(value.upper().replace("−", "-").split())
    if "O6MEG" in compact or "O6-MEG" in compact:
        raise APIError(f"{field}: O6-MeG is not supported by this canonical-DNA workflow; it will not be substituted with G.")


def safe_gmx(value: str) -> str:
    candidate = value.strip()
    if PurePosixPath(candidate.replace("\\", "/")).name not in {"gmx", "gmx_mpi"}:
        raise APIError("gmx must be gmx/gmx_mpi or an absolute path ending in one of those executable names.")
    return candidate


def wsl_argv(distribution: str, gmx: str, script: str, job: str, stage: str, extra: list[str]) -> list[str]:
    """Keep every WSL path as one argv item; this is important for Windows paths with spaces."""
    return ["wsl.exe", "-d", distribution, "--exec", DEFAULTS["wsl_python"], "-u", script, stage,
            "--job", job, "--gmx", gmx, *extra]


def wsl_error_text(data: bytes) -> str:
    """WSL warnings on this Windows host are UTF-16LE; keep them out of path stdout."""
    return data.decode("utf-16-le" if b"\0" in data else "utf-8", errors="replace").strip()


class App:
    def __init__(self):
        self.lock = threading.RLock()
        self.defaults = dict(DEFAULTS)
        self.environment = {
            "distribution": self.defaults["distribution"], "gmx": self.defaults["gmx"],
            "python": self.defaults["wsl_python"], "status": "not_checked", "error": None,
            "distributions": [], "version": None, "threading": self.defaults["threading_note"],
        }
        self.job: Path | None = None
        self.task: dict | None = None

    def state(self) -> dict:
        with self.lock:
            task = None if self.task is None else dict(self.task)
            environment = dict(self.environment)
            defaults = dict(self.defaults)
            job = self._job_state()
        return {"environment": environment, "job": job, "task": task, "defaults": defaults}

    def _job_state(self) -> dict | None:
        if self.job is None:
            return None
        config_path = self.job / "workflow.json"
        if not config_path.is_file():
            return None
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"path": str(self.job), "config": None, "stages": {}, "artifacts": {}, "error": str(exc)}
        complete = lambda name: (self.job / f"{name}.complete.json").is_file()
        stages = {name: "complete" if complete(name) else "pending" for name in ("topology", "solvate", "handoff")}
        md_stages = {}
        for stage in MD_STAGES:
            previous = {"em": "solvate", "nvt": "em", "npt": "nvt", "md": "npt"}[stage]
            inputs = ["topol.top", f"{stage}.mdp", {"em": "solv_ions.gro", "nvt": "em.gro", "npt": "nvt.gro", "md": "npt.gro"}[stage]]
            if stage in {"nvt", "npt"}:
                inputs.append("em.gro")
            if stage in {"npt", "md"}:
                inputs.append(f"{previous}.cpt")
            missing = [n for n in dict.fromkeys(inputs) if not (self.job / n).is_file()]
            ready_previous = complete("solvate") if stage == "em" else verified_return(self.job, previous)
            if not ready_previous:
                missing.append("先完成溶剂化" if stage == "em" else f"请核验{previous.upper()}回传结果")
            missing_returns = [n for n in [*return_files(stage), f"{stage}.mdp"] if not (self.job / n).is_file()]
            verified = verified_return(self.job, stage)
            tpr = self.job / f"{stage}.tpr"
            prepared = tpr.is_file() and tpr.stat().st_size > 0
            returned = any((self.job / f"{stage}.{ext}").is_file() for ext in ("gro", "log", "cpt", "xtc"))
            incomplete_return = returned and bool(missing_returns)
            status = "verified" if verified else ("blocked" if incomplete_return else ("returned" if returned else ("prepared" if prepared else ("blocked" if missing else "ready"))))
            md_stages[stage] = {"status": status, "can_prepare": not missing and not prepared,
                                "can_verify": not missing_returns, "missing_inputs": missing,
                                "missing_returns": missing_returns, "return_files": return_files(stage),
                                "command": f"gmx mdrun -deffnm {stage}", "tpr": str(tpr),
                                "predecessor": previous, "reason": ("回传不完整，缺少：" + "、".join(missing_returns)) if incomplete_return else "；".join(missing)}
            stages[stage] = "verified" if verified else ("unverified" if returned else ("prepared" if prepared else "pending"))
            stages[f"prepare_{stage}"] = "complete" if prepared else "pending"
            stages[f"verify_{stage}"] = "complete" if verified else "pending"
        artifacts = {}
        for name in ("workflow.json", "topology.pdb", "topol.top", "md.xtc", "dna.pdb", "dna_whole.xtc", "dna.tpr", "vmd_handoff.tcl", *(f"{s}.{ext}" for s in MD_STAGES for ext in ("tpr", "gro", "cpt", "log"))):
            path = self.job / name
            if path.exists():
                artifacts[name] = str(path)
        identity_file = "dna_manifest.json" if config.get("md_model", MD_MODELS[0]) != MD_MODELS[0] else config.get("forcefield", "")
        missing_trajectory = [n for n in ("md.tpr", "md.xtc", "topology.pdb", identity_file) if not (self.job / n).exists()]
        existing_trajectory = [n for n in ("dna.ndx", "dna.tpr", "dna_molecules.xtc", "dna_clustered.xtc", "dna_whole.xtc", "dna.pdb", "handoff.complete.json") if (self.job / n).exists()]
        try:
            centered = json.loads((self.job / "handoff.complete.json").read_text(encoding="utf-8")).get("centered", False)
        except (OSError, ValueError):
            centered = False
        trajectory = {"missing": missing_trajectory, "existing": existing_trajectory,
                      "can_process": not missing_trajectory and not existing_trajectory,
                      "centered": centered, "complete": centered and all(n in artifacts for n in ("dna.pdb", "dna_whole.xtc", "dna.tpr"))}
        return {"path": str(self.job), "config": config, "stages": stages, "md_stages": md_stages, "artifacts": artifacts, "trajectory": trajectory,
                "production_md": "verified" if verified_return(self.job, "md") else "生产MD完成情况未验证",
                "dos": self.dos_state()}

    def dos_state(self) -> dict:
        if self.job is None:
            return {}
        path = self.job / "dos_v2" / "workflow.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return {"error": str(exc)}
        return {"source": str(self.job / "gaussian_input")}

    def save_dos_state(self, state: dict) -> None:
        path = self.require_job() / "dos_v2" / "workflow.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def open_job(self, payload: dict) -> None:
        with self.lock:
            if self.task and self.task['status'] == 'running' and threading.current_thread().name != 'dna-gui-build':
                raise APIError('Wait for the running task before switching jobs.', 409)
        path = Path(required(payload, "path")).expanduser().resolve()
        if not (path / "workflow.json").is_file():
            raise APIError("Selected directory does not contain workflow.json.")
        try:
            json.loads((path / "workflow.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise APIError(f"Cannot read workflow.json: {exc}") from None
        with self.lock:
            if not self.task or self.task.get("status") != "running":
                self.task = None
            self.job = path

    def environment_check(self, payload: dict) -> None:
        distribution = payload.get("distribution", self.environment["distribution"])
        gmx = payload.get("gmx", self.environment["gmx"])
        if not isinstance(distribution, str) or not distribution.strip():
            raise APIError("distribution must be a non-empty WSL distribution name.")
        distribution = distribution.strip()
        gmx = safe_gmx(gmx) if isinstance(gmx, str) else self.environment["gmx"]
        try:
            listed = subprocess.run(["wsl.exe", "--list", "--quiet"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
            listed_text = listed.stdout.decode("utf-16-le", errors="replace").lstrip("\ufeff")
            distributions = [line.strip() for line in listed_text.splitlines() if line.strip()]
            status = subprocess.run(["wsl.exe", "--status"], text=True, encoding="utf-8", errors="replace",
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
            if listed.returncode or distribution not in distributions:
                raise RuntimeError(f"WSL distribution {distribution!r} is unavailable. {wsl_error_text(listed.stderr) or listed_text.strip()}")
            version = subprocess.run(["wsl.exe", "-d", distribution, "--exec", gmx, "--version"], text=True,
                                     encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            if version.returncode:
                raise RuntimeError(version.stdout.strip() or f"{gmx} --version failed")
            model = payload.get("md_model", MD_MODELS[0])
            if model not in MD_MODELS:
                raise RuntimeError("Unknown MD model")
            if model != MD_MODELS[0]:
                amber_python = payload.get("amber_python") or self.defaults["amber_python"]
                tleap = payload.get("amber_tleap") or self.defaults["amber_tleap"]
                check = subprocess.run(["wsl.exe", "-d", distribution, "--exec", amber_python, "-c",
                    "import sys, pathlib, parmed; p=pathlib.Path(sys.argv[1]); "
                    "assert p.is_file(), 'tleap missing'; "
                    "r=p.parent.parent/'dat/leap/cmd'; "
                    "assert (r/('leaprc.DNA.'+sys.argv[2])).is_file(), 'DNA parameters missing'; "
                    "assert (r/'leaprc.water.opc').is_file(), 'OPC parameters missing'; "
                    "print('ParmEd '+parmed.__version__+'; '+sys.argv[2]+' + OPC available')",
                    tleap, model.split('_')[0].upper()], text=True, encoding="utf-8", errors="replace",
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                if check.returncode:
                    raise RuntimeError(check.stdout.strip() or "AmberTools environment check failed")
                version.stdout += "\n" + check.stdout
            with self.lock:
                self.environment.update({"distribution": distribution, "gmx": gmx, "status": "ready", "error": None,
                                         "distributions": distributions, "version": version.stdout.strip(), "wsl_status": status.stdout.strip()})
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            with self.lock:
                self.environment.update({"distribution": distribution, "gmx": gmx, "status": "unavailable", "error": str(exc)})

    def require_job(self) -> Path:
        with self.lock:
            if self.job is None:
                raise APIError("Open an existing job or create a new one before this action.", 409)
            return self.job

    def _append(self, task_id: str, text: str) -> None:
        with self.lock:
            if self.task is None or self.task["id"] != task_id:
                return
            self.task["log"] = (self.task["log"] + text)[-MAX_LOG:]

    def _run(self, task_id: str, argv: list[str], cwd: Path = HERE) -> None:
        self._append(task_id, "$ " + subprocess.list2cmdline(argv) + "\n")
        proc = subprocess.Popen(argv, cwd=cwd, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append(task_id, line)
        code = proc.wait()
        if code:
            raise RuntimeError(f"Command failed with exit code {code}.")

    def _wsl_path(self, distribution: str, path: Path) -> str:
        proc = subprocess.run(["wsl.exe", "-d", distribution, "--exec", "wslpath", "-a", "-u", str(path)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        value = proc.stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode or not value:
            raise RuntimeError(f"Cannot convert Windows path for WSL: {wsl_error_text(proc.stderr)}")
        return value

    def _wsl_command(self, job: Path, stage: str, payload: dict) -> list[str]:
        distribution = payload.get("distribution", self.environment["distribution"])
        gmx = payload.get("gmx", self.environment["gmx"])
        if not isinstance(distribution, str) or not distribution.strip():
            raise APIError("distribution must be a WSL distribution name.")
        gmx = safe_gmx(gmx) if isinstance(gmx, str) else self.environment["gmx"]
        distribution = distribution.strip()
        script = self._wsl_path(distribution, HERE / "md_pipeline.py")
        job_path = self._wsl_path(distribution, job)
        extra = []
        if stage == "topology":
            extra.append("--noninteractive-terminals")
        if stage in {"em", "nvt", "npt", "md"} and PurePosixPath(gmx).name == "gmx":
            extra += ["--ntmpi", "1"]
        if stage in {"em", "nvt", "npt"} or payload.get("ntomp") is not None:
            ntomp = int(payload.get("ntomp", self.defaults["ntomp"]))
            if ntomp < 1:
                raise APIError("ntomp must be positive.")
            extra += ["--ntomp", str(ntomp)]
        return wsl_argv(distribution, gmx, script, job_path, stage, extra)

    def _start(self, action: str, work) -> None:
        with self.lock:
            if self.task and self.task["status"] == "running":
                raise APIError("Another task is already running; wait for it before starting a second one.", 409)
            task_id = str(uuid.uuid4())
            self.task = {"id": task_id, "action": action, "status": "running", "log": "", "error": None}

        def runner():
            try:
                work(task_id)
            except Exception as exc:  # The state API must show a clear failure, not infer success from files.
                self._append(task_id, f"ERROR: {exc}\n")
                with self.lock:
                    if self.task and self.task["id"] == task_id:
                        self.task.update(status="failed", error=str(exc))
            else:
                with self.lock:
                    if self.task and self.task["id"] == task_id:
                        self.task.update(status="succeeded")

        threading.Thread(target=runner, name=f"dna-gui-{action}", daemon=True).start()

    def create_job(self, payload: dict) -> None:
        sequence = required(payload, "sequence")
        reject_o6meg(sequence, "sequence")
        mode = required(payload, "mode")
        if mode not in {"single", "duplex", "partner"}:
            raise APIError("mode must be single, duplex, or partner.")
        partner = payload.get("partner")
        if mode == "partner":
            partner = required(payload, "partner")
            reject_o6meg(partner, "partner")
        output = Path(required(payload, "output")).expanduser().resolve()
        model = payload.get("md_model", MD_MODELS[0])
        if model not in MD_MODELS:
            raise APIError("Unsupported DNA/water model.")
        amber = model != MD_MODELS[0]
        forcefield = None if amber else Path(required(payload, "forcefield")).expanduser().resolve()
        if not amber:
            if not forcefield.is_dir() or not (forcefield / "na.rtp").is_file():
                raise APIError(f"CHARMM 力场目录无效或缺少 na.rtp：{forcefield}")
        if amber:
            for key in ("amber_python", "amber_tleap"):
                value = payload.get(key, self.defaults[key])
                if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\n\r\0"):
                    raise APIError(f"{key} must be an absolute WSL executable path.")
            if payload.get("salt_cation", "Na+") not in {"Na+", "K+"}:
                raise APIError("Select Na+ or K+ for OPC salt.")
        if output.exists():
            raise APIError("output already exists; choose a new empty job directory rather than overwriting a prior job.")
        values = {name: number(payload, name) for name in ("temperature", "salt", "nvt_ps", "npt_ps", "md_ns", "xtc_ps", "padding", "seed")}
        box_type = payload.get("box_type", "dodecahedron")
        if box_type not in {"cubic", "dodecahedron", "octahedron", "triclinic"}:
            raise APIError("Unsupported box type.")
        if amber and box_type == "dodecahedron":
            raise APIError("AmberTools 首版不支持菱形十二面体；请明确选择 cubic、triclinic 或 octahedron（截角八面体）。")
        if not RUNTIME.is_file():
            raise APIError(f"Workflow runtime was not found: {RUNTIME}")

        def work(task_id: str):
            argv = [str(RUNTIME), "-u", str(HERE / "dna_workflow.py"), "build", "--sequence", sequence,
                    "--md-model", model, "--output", str(output), "--temperature", str(values["temperature"]),
                    "--salt", str(values["salt"]), "--nvt-ps", str(values["nvt_ps"]), "--npt-ps", str(values["npt_ps"]),
                    "--md-ns", str(values["md_ns"]), "--xtc-ps", str(values["xtc_ps"]), "--padding", str(values["padding"]),
                    "--seed", str(int(values["seed"])), "--box-type", box_type]
            if amber:
                argv += ["--amber-python", payload.get("amber_python", self.defaults["amber_python"]),
                         "--amber-tleap", payload.get("amber_tleap", self.defaults["amber_tleap"]),
                         "--salt-cation", payload.get("salt_cation", "Na+")]
            else:
                argv += ["--forcefield", str(forcefield)]
            argv += ["--single"] if mode == "single" else (["--duplex"] if mode == "duplex" else ["--partner", partner])
            self._run(task_id, argv)
            self.open_job({"path": str(output)})

        self._start("build", work)

    def action(self, payload: dict) -> None:
        action = required(payload, "action")
        if action not in ALLOWED_ACTIONS:
            raise APIError(f"Unsupported action: {action}.")
        job = self.require_job()
        if action in DOS_ACTIONS:
            self.dos_action(action, job, payload)
            return
        if action in ANALYSIS_ACTIONS:
            self.analysis_action(action, job, payload)
            return
        if action == "gaussian":
            if payload.get("gaussian_job"):
                job = Path(required(payload, "gaussian_job")).resolve()
                if not (job / "workflow.json").is_file():
                    raise APIError("Gaussian source job must contain workflow.json.")
            frame = Path(required(payload, "frame")).expanduser().resolve()
            output = Path(required(payload, "output")).expanduser().resolve()
            if output.exists():
                raise APIError("Gaussian output already exists; choose a new output directory.")
            if not RUNTIME.is_file():
                raise APIError(f"Workflow runtime was not found: {RUNTIME}")
        if action == "handoff":
            config = json.loads((job / "workflow.json").read_text(encoding="utf-8"))
            identity_file = "dna_manifest.json" if config.get("md_model", MD_MODELS[0]) != MD_MODELS[0] else config.get("forcefield", "")
            missing = [name for name in ("topology.pdb", "md.tpr", "md.xtc", identity_file) if not (job / name).exists()]
            if missing:
                raise APIError("handoff requires workflow/forcefield/topology.pdb/md.tpr/md.xtc; missing: " + ", ".join(missing))
        if action == "open_vmd":
            missing = [name for name in ("dna.pdb", "dna_whole.xtc") if not (job / name).is_file()]
            if missing:
                raise APIError("open_vmd requires dna.pdb and dna_whole.xtc: " + ", ".join(missing))

        def work(task_id: str):
            if action in {"topology", "solvate", "handoff"} | PREPARATION_ACTIONS | VERIFICATION_ACTIONS:
                argv = self._wsl_command(job, action, payload)
                if action == "handoff":
                    self._append(task_id, "Server material: workflow.json, copied force field, topology.pdb, md.tpr, md.xtc.\n")
                    self._append(task_id, "Handoff command: " + subprocess.list2cmdline(argv) + "\n")
                    self._append(task_id, "This post-processing action does not verify that production MD reached 100 ns.\n")
                self._run(task_id, argv, job)
            elif action == "gaussian":
                argv = [str(RUNTIME), "-u", str(HERE / "dna_workflow.py"), "gaussian", "--job", str(job), "--frame", str(frame),
                        "--output", str(output), "--time-ps", str(payload.get("time_ps", "")),
                        "--nproc", str(int(payload.get("nproc", 16))), "--memory", str(payload.get("memory", "32GB")),
                        "--dispersion", str(payload.get("dispersion", "none"))]
                if payload.get("time_ps") in (None, ""):
                    argv.remove("--time-ps")
                    argv.remove("")
                self._run(task_id, argv)
            elif action == "open_folder":
                os.startfile(str(job))
                self._append(task_id, f"Opened {job}\n")
            else:  # open_vmd
                if not VMD.is_file():
                    raise RuntimeError(f"VMD was not found: {VMD}")
                subprocess.Popen([str(VMD), "-e", str(HERE / "vmd_handoff.tcl")], cwd=job)
                self._append(task_id, "Opened VMD with vmd_handoff.tcl; its PDB frame is removed before dna_whole.xtc loads.\n")

        self._start(action, work)

    def dos_action(self, action: str, job: Path, payload: dict) -> None:
        """Use the scientific CLI; MD/VMD status and local CUDA are not prerequisites."""
        state = self.dos_state()
        if action == "dos_open":
            folder = Path(required(payload, "path")).resolve()
            if not folder.is_dir():
                raise APIError("找不到此目录。")
            os.startfile(str(folder))
            return
        python = Path(payload.get("dos_python") or self.defaults["dos_python"])
        if not python.is_file():
            raise APIError(f"DoS Python 不存在：{python}。请选择安装 NumPy/SciPy/Matplotlib 的 Python。")
        base = [str(python), "-X", "utf8", "-B", "-u", str(DOS_SCRIPT)]
        inspection = job / "dos_v2" / "inspection"
        if action == "dos_inspect":
            source = Path(required(payload, "source")).resolve()
            if not source.is_dir():
                raise APIError("请选择 Gaussian 矩阵目录。")
            argv = [*base, "inspect", "--source", str(source), "--output", str(inspection)]
        elif action == "dos_export":
            if not (inspection / "inspection.json").is_file():
                raise APIError("请先检查 Gaussian 矩阵和原子映射。")
            inspected = json.loads((inspection / "inspection.json").read_text(encoding="utf-8"))
            if Path(required(payload, "source")).resolve() != Path(inspected["source"]):
                raise APIError("输入目录已改变，请重新检查矩阵。")
            output = Path(required(payload, "output")).resolve()
            if output.exists():
                raise APIError("计算包目录已存在。请为新参数或新构型选择新的目录。")
            argv = [*base, "export", "--inspection", str(inspection), "--output", str(output)]
            parameters = payload.get("parameters", {})
            if not isinstance(parameters, dict):
                raise APIError("parameters must be an object.")
            for key in DOS_PARAMETERS:
                if key in parameters:
                    argv += ["--" + key.replace("_", "-"), str(number(parameters, key))]
        else:
            results = Path(required(payload, "results")).resolve()
            if not (results / "run_manifest.json").is_file():
                raise APIError("回传目录缺少 run_manifest.json。请复制完整的 results_c64 目录。")
            package_value = payload.get("package") or state.get("package")
            package = Path(package_value).resolve() if package_value else None
            if package is not None and not (package / "manifest.json").is_file():
                raise APIError("所选计算包缺少 manifest.json。")
            verify = [*base, "verify", "--results", str(results)]
            if package:
                verify += ["--package", str(package)]
            if action == "dos_verify":
                argv = verify
            else:
                output = Path(required(payload, "output")).resolve()
                if output.exists():
                    raise APIError("图件目录已存在。请选择新的目录以保留已有图件。")
                argv = [*base, "plot", "--results", str(results), "--output", str(output)]
                for key in ("gateway", "gateway_min", "gateway_max", "temperature", "display_sigma"):
                    if payload.get(key) not in (None, ""):
                        argv += ["--" + key.replace("_", "-"), str(number(payload, key))]

        def work(task_id):
            if action == "dos_plot":
                self._run(task_id, verify, job)
            self._run(task_id, argv, job)
            current = self.dos_state()
            current["dos_python"] = str(python)
            if action == "dos_inspect":
                report = json.loads((inspection / "inspection.json").read_text(encoding="utf-8"))
                if current.get("inspection", {}).get("source_files") != report["source_files"]:
                    current = {"dos_python": str(python)}
                current.update(source=str(source), inspection=report)
            elif action == "dos_export":
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                current.update(package=str(output), package_phase=manifest["phase"], parameters=manifest["parameters"],
                               run_id=manifest["run_id"], results="", verification=None, figure=None)
            elif action == "dos_verify":
                report = json.loads((results / "verification_report.json").read_text(encoding="utf-8"))
                if current.get("results") != str(results):
                    current["figure"] = None
                current.update(results=str(results), verification=report)
                if package is not None:
                    current["package"] = str(package)
            else:
                report = json.loads((output / "figure_metadata.json").read_text(encoding="utf-8"))
                current.update(results=str(results), figure=report | {"output": str(output)},
                               verification=json.loads((results / "verification_report.json").read_text(encoding="utf-8")))
            self.save_dos_state(current)
            with self.lock:
                if self.task and self.task["id"] == task_id:
                    self.task["result"] = {"kind": action, "dos": current}
            if action == "dos_export":
                self._append(task_id, "计算包已生成。将整个目录复制到超算；本机进入等待回传状态。\n")

        self._start(action, work)

    def analysis_action(self, action, job, payload):
        """Thin CLI adapters. Each analysis owns a new output directory, not the MD job."""
        if action in {"cluster", "open_tk"}:
            missing = [n for n in ("dna.pdb", "dna_whole.xtc") if not (job / n).is_file()]
            if missing:
                raise APIError("先处理MD轨迹；缺少: " + ", ".join(missing))
        if action not in {"verify_relax", "open_tk"}:
            output = Path(required(payload, "output")).resolve()
            if output.exists():
                raise APIError("输出目录已存在；请选择新的独立目录，避免覆盖。")
        else:
            output = Path(required(payload, "child_job")).resolve() if action == "verify_relax" else job
        if action == "prepare_relax":
            frame = Path(required(payload, "frame")).resolve()
            if not frame.is_file():
                raise APIError("找不到代表构型PDB。")

        def finish(task_id, result):
            with self.lock:
                if self.task and self.task["id"] == task_id:
                    self.task["result"] = result
            self._append(task_id, "输出: " + str(output) + "\n")

        def work(task_id):
            if action == "open_tk":
                if not VMD.is_file():
                    raise FileNotFoundError(VMD)
                subprocess.Popen([str(VMD), "-e", str(HERE / "vmd_open_tk.tcl")], cwd=job)
                self._append(task_id, "已启动独立VMD窗口与原生Tk控制台；不改变现有VMD会话。\n")
                return
            if action == "cluster":
                base = [str(RUNTIME), "-X", "utf8", "-u", str(HERE / "vmd_workflow.py")]
                argv = [*base, "cluster", "--job", str(job), "--output", str(output), "--vmd", str(VMD)]
                values = {"algorithm": payload.get("algorithm", "gromos"),
                          "selection": payload.get("selection", "not hydrogen"), "distfunc": payload.get("distfunc", "fitrmsd"),
                          "cutoff": payload.get("cutoff", 2.5), "num": payload.get("num", 5),
                          "first": payload.get("first", 0), "last": payload.get("last", -1), "step": payload.get("step", 10)}
                for key,value in values.items():
                    argv += ["--"+key, str(value)]
                self._run(task_id, argv, job)
                finish(task_id, {"output": str(output), "kind": action})
                return
            distro = payload.get("distribution", self.environment["distribution"])
            gmx = safe_gmx(payload.get("gmx", self.environment["gmx"]))
            selected_job = output if action == "verify_relax" else job
            argv = ["wsl.exe", "-d", distro, "--exec", DEFAULTS["wsl_python"], "-u",
                    self._wsl_path(distro, HERE / "md_extensions.py"), action,
                    "--job", self._wsl_path(distro, selected_job), "--gmx", gmx]
            if action != "verify_relax":
                argv += ["--output", self._wsl_path(distro, output), "--padding", str(payload.get("padding", 1.5))]
            if action == "prepare_relax":
                argv += ["--frame", self._wsl_path(distro, frame), "--salt", str(payload.get("salt", 0.15)),
                         "--box-type", str(payload.get("box_type", "dodecahedron"))]
            self._run(task_id, argv, job)
            result = {"output": str(output), "kind": action}
            if action == "verify_relax":
                result["frame"] = str(output / "relaxed_dna.pdb")
                result["gaussian_job"] = str(output)
            finish(task_id, result)

        self._start(action, work)

    def browse(self, payload: dict) -> str:
        kind = required(payload, "kind")
        if kind not in {"file", "directory"}:
            raise APIError("kind must be file or directory.")
        picker = """import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
path = filedialog.askopenfilename() if sys.argv[1] == 'file' else filedialog.askdirectory()
print(path)
root.destroy()
"""
        try:
            proc = subprocess.run([sys.executable, "-c", picker, kind], text=True, encoding="utf-8", errors="replace",
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise APIError(f"File picker could not start: {exc}") from None
        if proc.returncode:
            raise APIError(proc.stderr.strip() or "File picker failed.")
        return proc.stdout.strip()


class Handler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def log_message(self, format, *args):
        return

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def send_json(self, payload: dict, status=200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self.send_json(self.app.state())
            return
        if path == "/api/dos/preview":
            figure = self.app.dos_state().get("figure") or {}
            preview = Path(figure.get("preview", ""))
            if not preview.is_file() or preview.suffix.lower() != ".png":
                self.send_json({"error": "尚无图件预览。"}, 404)
                return
            body = preview.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/"):
            self.send_json({"error": "Not found."}, 404)
            return
        relative = "index.html" if path in {"", "/"} else unquote(path).lstrip("/")
        root = GUI.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self.send_json({"error": "Not found."}, 404)
            return
        if not target.is_file():
            self.send_json({"error": "GUI file not found."}, 404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Origin") != expected:
            self.send_json({"error": "POST requests must originate from this local GUI."}, 403)
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json":
            self.send_json({"error": "POST body must be application/json."}, 415)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 1_000_000:
                raise APIError("Invalid JSON body length.")
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                raise APIError("JSON body must be an object.")
            path = urlparse(self.path).path
            if path == "/api/environment":
                self.app.environment_check(payload)
                response = self.app.state()
            elif path == "/api/job/create":
                self.app.create_job(payload)
                response = self.app.state()
            elif path == "/api/job/open":
                self.app.open_job(payload)
                response = self.app.state()
            elif path == "/api/action":
                self.app.action(payload)
                response = self.app.state()
            elif path == "/api/browse":
                response = {"path": self.app.browse(payload)}
            else:
                raise APIError("Not found.", 404)
        except APIError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Malformed JSON."}, 400)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)
        else:
            self.send_json(response)


def make_server(host="127.0.0.1", port=8765, app: App | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler, bind_and_activate=False)
    if os.name == "nt":
        # Windows SO_REUSEADDR otherwise permits old/new GUI processes on the same port.
        server.allow_reuse_address = False
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        server.server_bind()
        server.server_activate()
    except OSError:
        server.server_close()
        raise
    server.app = app or App()  # type: ignore[attr-defined]
    server.app.defaults.update(host=host, port=server.server_port)  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535:
        parser.error("Only 127.0.0.1 and a valid TCP port are supported.")
    server = make_server(args.host, args.port)
    print(f"DNA workflow GUI: http://{args.host}:{args.port}", flush=True)
    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
