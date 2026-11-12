"""`bitino fetch`: download a model's ORIGINAL weights and code, verify them, convert locally.

The recipe (registry/recipes.json, built by tools/build_recipes.py) is data: downloads with SHA-256, source
trees pinned to a commit, the reference environment and the converter command. This module only executes it:

  1. every download goes into a new, empty directory  <home>/downloads/<model-id>/  and is hashed while it
     streams; a SHA-256 mismatch deletes the file and stops;
  2. source trees are GitHub archives of the recorded commit (or a PyPI wheel), extracted with tar/zip member
     checks into <home>/src/ (read only afterwards);
  3. the converter runs as a subprocess with ``python -I`` (isolated: no user site, no PYTHONPATH, no current
     directory on sys.path) in an empty work directory; checkpoints are only ever read with
     ``torch.load(weights_only=True)`` by the converters;
  4. the converted file's data section is compared with the recorded data_sha256: equal means the weights and
     constants are bit-identical to the published conversion (the JSON manifest differs: it records the date
     and library versions of *your* conversion).

The converters also produce reference outputs on an evaluation set. Datasets are not redistributed, so unless
``--eval-dir`` points at a real eval set, a small synthetic one (seeded random inputs with the model's input
shape and preprocessing key) is generated: conversion and runtime-vs-numpy checks work the same, accuracy
numbers are meaningless on it.
"""
from __future__ import annotations

import datetime
import hashlib
import html
import http.cookiejar
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from . import zoo

UA = "bitino-fetch/1 (+https://github.com/fedeloper/bitino)"


class FetchError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[fetch] {msg}", file=sys.stderr, flush=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _header(path: str) -> dict:
    with open(path, "rb") as f:
        hdr = f.read(64)
        if hdr[:8] != b"BFMODEL1":   # format v1 magic (unchanged since the format was introduced)
            return {}
        jo, js = struct.unpack("<QQ", hdr[16:32])
        f.seek(jo)
        return json.loads(f.read(js))


def data_sha256(path: str) -> str | None:
    return _header(path).get("data_sha256")


def graph_sha256(path: str) -> str:
    """Hash of the graph (constants table, inputs, nodes, outputs): everything except the provenance manifest."""
    h = _header(path)
    g = json.dumps({k: h.get(k) for k in ("tensors", "inputs", "nodes", "outputs")}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(g.encode()).hexdigest()


_DRIVE_ID = re.compile(r"(?:[?&]id=|/file/d/)([\w-]{20,})")


def drive_file_id(url: str) -> str | None:
    """File id of a Google Drive file link (uc?id=, open?id=, /file/d/<id>/view, drive.usercontent), else None."""
    if "drive.google.com" not in url and "drive.usercontent.google.com" not in url:
        return None
    m = _DRIVE_ID.search(url)
    return m.group(1) if m else None


def direct_url(url: str) -> str:
    """The URL that serves the file's bytes: Google Drive file links become the drive.usercontent download
    endpoint (confirm=t skips the "can't scan for viruses" page of files over ~100 MB); others are unchanged."""
    fid = drive_file_id(url)
    if fid:
        return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    return url


class _HTMLPage(Exception):
    def __init__(self, text: str):
        super().__init__("HTML page")
        self.text = text


def _drive_form(text: str) -> str | None:
    """Google Drive's download-confirmation page: the URL its form submits to, else None."""
    m = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"[^>]*>(.*?)</form>', text, re.S)
    if not m:
        return None
    fields = dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', m.group(2)))
    if "id" not in fields:
        return None
    return html.unescape(m.group(1)) + "?" + urllib.parse.urlencode(fields)


# Session cookies only (no stored credentials): OneDrive/SharePoint share links set an access cookie on the first
# redirect and serve the file on the next request.
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def _stream(url: str, tmp: str) -> tuple[str, int]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    h = hashlib.sha256()
    n = 0
    with _OPENER.open(req, timeout=120) as r, open(tmp, "wb") as f:
        ctype = r.headers.get("Content-Type", "")
        first = True
        for chunk in iter(lambda: r.read(1 << 20), b""):
            if first and ctype.startswith("text/html") and (b"<html" in chunk[:4096].lower() or b"<!doctype html" in chunk[:4096].lower()):
                raise _HTMLPage((chunk + r.read(1 << 20)).decode("utf-8", "replace"))
            first = False
            h.update(chunk)
            f.write(chunk)
            n += len(chunk)
    return h.hexdigest(), n


BYTES_DOWNLOADED = [0]   # running total of this process (reported by fetch records)


def download(url: str, dest: str, sha256: str | None, retries: int = 3) -> str:
    """Streams url to dest (a path inside a fresh directory), verifying SHA-256 if given. Returns the hash.

    Transient failures (timeouts, connection resets, HTTP 429/5xx) are retried with backoff. Google Drive links
    go through direct_url() and its confirmation form; Drive's quota / sign-in pages raise a FetchError that says
    how to download the file by hand."""
    if os.path.exists(dest):
        got = sha256_file(dest)
        if not sha256 or got == sha256:
            log(f"cached {os.path.basename(dest)} ({got[:12]})")
            return got
        os.remove(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    src = direct_url(url)
    log(f"download {url}" + (f" (via {src})" if src != url else ""))
    tmp = dest + ".part"
    got = None
    for attempt in range(retries):
        try:
            try:
                got, n = _stream(src, tmp)
            except _HTMLPage as page:
                nxt = _drive_form(page.text) if drive_file_id(url) else None
                if not nxt:
                    raise
                log("Google Drive confirmation page; following its download form")
                got, n = _stream(nxt, tmp)
            BYTES_DOWNLOADED[0] += n
            break
        except _HTMLPage as page:
            if os.path.exists(tmp):
                os.remove(tmp)
            low = page.text.lower()
            if drive_file_id(url) and ("quota" in low or "too many users" in low):
                why = "Google Drive refused the download (the file's daily download quota is exhausted)"
            elif drive_file_id(url) and ("accounts.google.com" in low or "sign in" in low):
                why = "Google Drive asks for a sign-in (the file is not shared publicly any more)"
            else:
                why = f"{url} returned an HTML page (login wall or confirmation page)"
            raise FetchError(f"{why}. Open {url} in a browser, download the file, then pass --file NAME=PATH "
                             f"(the SHA-256 is still checked)")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            code = getattr(e, "code", None)
            if os.path.exists(tmp):
                os.remove(tmp)
            if (code is not None and code not in (408, 429) and code < 500) or attempt == retries - 1:
                raise FetchError(f"download of {url} failed: {e}") from e
            wait = 10 * 3 ** attempt
            log(f"download failed ({e}); retrying in {wait} s")
            time.sleep(wait)
    if sha256 and got != sha256.lower():
        os.remove(tmp)
        raise FetchError(f"SHA-256 mismatch for {url}: expected {sha256}, got {got} (file deleted)")
    os.replace(tmp, dest)
    if not sha256:
        log(f"WARNING: no recorded SHA-256 for {os.path.basename(dest)}; got {got}")
    return got


def extract_member(archive: str, member: str, dest: str, sha256: str | None) -> str:
    """One file of a zip archive (a release that bundles several models) -> dest, SHA-256 checked."""
    with zipfile.ZipFile(archive) as z:
        _safe_members([member])
        with z.open(member) as src, open(dest + ".part", "wb") as f:
            shutil.copyfileobj(src, f, 1 << 20)
    got = sha256_file(dest + ".part")
    if sha256 and got != sha256.lower():
        os.remove(dest + ".part")
        raise FetchError(f"SHA-256 mismatch for {member} in {os.path.basename(archive)}: expected {sha256}, got {got}")
    os.replace(dest + ".part", dest)
    return got


def _safe_members(names: list[str]) -> None:
    for n in names:
        if n.startswith("/") or ".." in n.replace("\\", "/").split("/"):
            raise FetchError(f"unsafe archive member {n!r}")


def extract_tar_strip(archive: str, dest: str) -> None:
    tmp = tempfile.mkdtemp(prefix=".extract-", dir=os.path.dirname(dest))
    try:
        with tarfile.open(archive) as t:
            _safe_members(t.getnames())
            members = [m for m in t.getmembers() if m.isfile() or m.isdir()]   # no links, devices or fifos
            try:
                t.extractall(tmp, members=members, filter="data")
            except TypeError:   # Python < 3.12
                t.extractall(tmp, members=members)
        tops = os.listdir(tmp)
        root = os.path.join(tmp, tops[0]) if len(tops) == 1 and os.path.isdir(os.path.join(tmp, tops[0])) else tmp
        shutil.move(root, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract_zip(archive: str, dest: str) -> None:
    with zipfile.ZipFile(archive) as z:
        _safe_members(z.namelist())
        z.extractall(dest)


def get_source(src: dict, home: str) -> str:
    kind = src.get("kind")
    sdir = os.path.join(home, "src")
    if kind == "github-archive":
        name = src["repo"].rstrip("/").split("github.com/")[-1].replace("/", "-") + "-" + src["commit"][:12]
        dest = os.path.join(sdir, name)
        if not os.path.isdir(dest) and shutil.which("git"):
            # A git checkout of the exact commit: converters record the commit from .git/HEAD.
            tmp = dest + ".partial"
            shutil.rmtree(tmp, ignore_errors=True)
            log(f"git clone {src['repo']} @ {src['commit']}")
            env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
            r = subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", src["repo"] + ".git", tmp], env=env)
            if r.returncode == 0:
                r = subprocess.run(["git", "-C", tmp, "-c", "advice.detachedHead=false", "checkout", "--quiet", src["commit"]], env=env)
            if r.returncode == 0:
                full = subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
                if not full.startswith(src["commit"]):
                    raise FetchError(f"{src['repo']}: checked out {full}, expected {src['commit']}")
                with open(os.path.join(tmp, "COMMIT_SHA"), "w") as f:
                    f.write(full + "\n")
                os.replace(tmp, dest)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
                log("git clone failed; falling back to the GitHub archive of the commit")
        if not os.path.isdir(dest):
            arch = os.path.join(sdir, ".archives", name + ".tar.gz")
            download(src["url"], arch, src.get("sha256"))
            extract_tar_strip(arch, dest)
            with open(os.path.join(dest, "COMMIT_SHA"), "w") as f:
                f.write(src["commit"] + "\n")
        return os.path.join(dest, src.get("subdir") or "")
    if kind == "pypi-wheel":
        dest = os.path.join(sdir, f"{src['package']}-{src['version']}")
        if not os.path.isdir(dest):
            meta_url = f"https://pypi.org/pypi/{src['package']}/{src['version']}/json"
            with urllib.request.urlopen(urllib.request.Request(meta_url, headers={"User-Agent": UA}), timeout=60) as r:
                meta = json.load(r)
            f = next((u for u in meta["urls"] if u["filename"] == src["file"]), None)
            if not f:
                raise FetchError(f"{src['file']} not on PyPI")
            want = f["digests"]["sha256"]
            if src.get("sha256_prefix") and not want.startswith(src["sha256_prefix"]):
                raise FetchError(f"PyPI digest {want} does not match the recorded {src['sha256_prefix']}")
            arch = os.path.join(sdir, ".archives", src["file"])
            download(f["url"], arch, want)
            extract_zip(arch, dest)
            with open(os.path.join(dest, "COMMIT_SHA"), "w") as fh:
                fh.write(f"pypi {src['package']}=={src['version']} {src['file']} sha256:{want}\n")
        return dest
    raise FetchError(f"unknown source kind {kind!r}")


def resolve_python(env: str, overrides: dict) -> str:
    if env in overrides:
        return overrides[env]
    v = os.environ.get(f"BITINO_PY_{env.upper()}")
    if v:
        return v
    if env in ("base", "torch"):
        return sys.executable
    spec = zoo.recipe_doc().get("envs", {}).get(env, {})
    raise FetchError(f"this recipe needs the '{env}' reference environment ({spec.get('note', '')}). Create it, e.g.\n"
                     f"    python{spec.get('python', '3')} -m venv ~/.venvs/bitino-{env}\n"
                     f"    ~/.venvs/bitino-{env}/bin/pip install -r {spec.get('requirements') or '<see docs>'}\n"
                     f"then pass --python {env}=~/.venvs/bitino-{env}/bin/python (or set BITINO_PY_{env.upper()})")


def synthetic_eval(recipe: dict, home: str, n: int, seed: int = 0) -> str:
    """A tiny eval set of seeded random inputs (numpy only), with the manifest layout the converters read."""
    import numpy as np
    ev = recipe["eval"]
    key = ev["preprocess"]
    shape = list(ev.get("input_shape") or [])
    if not shape:
        raise FetchError("recipe has no input shape; pass --eval-dir")
    sets = zoo.eval_sets()
    pre_def = (sets.get(ev["dir"], {}).get("preprocessing", {}).get(key) or {})
    n_cls = sets.get(ev["dir"], {}).get("n_classes_covered") or 10
    d = os.path.join(home, "eval", f"synthetic-{key}-{'x'.join(map(str, shape[1:]))}-n{n}")
    man_p = os.path.join(d, "manifest.json")
    if os.path.exists(man_p):
        return d
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    extra_pre = {}
    if key == "bibert":   # token inputs: [CLS] random word-piece ids [SEP], one segment, full attention mask
        L = shape[1]
        ids = rng.integers(1000, 30000, size=(n, L), dtype=np.int32)
        ids[:, 0], ids[:, -1] = 101, 102
        ids.tofile(os.path.join(d, "input_ids.i32"))
        np.zeros((n, L), np.int32).tofile(os.path.join(d, "token_type_ids.i32"))
        np.ones((n, L), np.float32).tofile(os.path.join(d, "attention_mask.f32"))
        (np.arange(n) % 2).astype(np.int32).tofile(os.path.join(d, "labels.i32"))
        x = np.concatenate([ids.view(np.float32), np.zeros((n, L), np.int32).view(np.float32),
                            np.ones((n, L), np.float32)], axis=1)   # per-sample record, bench input order
    elif key == "tensorpack_bgr":   # uint8 BGR crops for the TF reference + the float input its graph computes
        u8 = rng.integers(0, 256, size=[n] + shape[1:], dtype=np.uint8)
        u8.tofile(os.path.join(d, key + "_u8.u8"))
        extra_pre["uint8_file"] = key + "_u8.u8"
        mean = np.asarray([0.406, 0.456, 0.485], np.float32) * np.float32(255.0)
        std = np.asarray([0.225, 0.224, 0.229], np.float32) * np.float32(255.0)
        x = ((u8.astype(np.float32) - mean) * (np.float32(1.0) / std)).astype(np.float32)   # tools/eval_sets/tensorpack_bgr.py
    elif key.endswith("_pixels"):   # raw 8-bit pixel values (FINN / QONNX graphs check the UINT8 input type)
        x = rng.integers(0, 256, size=[n] + shape[1:]).astype(np.float32)
    elif key.endswith("_totensor"):   # ToTensor(): k / 255
        x = (rng.integers(0, 256, size=[n] + shape[1:]) / 255.0).astype(np.float32)
    else:
        x = rng.standard_normal([n] + shape[1:]).astype(np.float32)
    x.tofile(os.path.join(d, key + ".f32"))
    man = {"schema": "bitino.eval_set.v1", "name": f"synthetic-{n}",
           "source": "seeded random inputs of the model's input type, written by bitino fetch (no dataset); accuracy is meaningless",
           "n_classes_covered": n_cls,
           "preprocessing": {key: {"shape": [n] + shape[1:], "dtype": "float32", "layout": pre_def.get("layout", "NHWC"),
                                   "file": key + ".f32", "sha256": sha256_file(os.path.join(d, key + ".f32")),
                                   "definition": pre_def.get("definition", key), **extra_pre}},
           "samples": [{"file": f"synthetic/{i}", "label": int(i % n_cls)} for i in range(n)],
           "limitations": "synthetic inputs: only for conversion and runtime-vs-reference checks"}
    with open(man_p, "w") as f:
        json.dump(man, f, indent=1)
    return d


def _fmt(arg: str, ctx: dict) -> str:
    out = arg
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", v)
    if "{" in out and "}" in out and not out.startswith("see "):
        raise FetchError(f"unresolved placeholder in {arg!r} (missing --file/--source?)")
    return out


def fetch_prebuilt(model_id: str) -> dict:
    """Downloads the converted files of a redistributable model from the GitHub release (SHA-256 checked)."""
    man = zoo._load(os.path.join("registry", "release_models.json"))
    ent, why = zoo.release_entry(model_id)
    if not ent:
        ok = [k for k in (man.get("models") or {})]
        raise FetchError(f"{model_id} has no prebuilt release file ({why}). Only models whose weights licence is "
                         f"explicitly permissive are redistributed: {', '.join(ok) or 'none'}. "
                         f"Use `bitino fetch {model_id}` to convert it from the original source.")
    out_dir = zoo.model_dir(model_id)
    base = f"https://github.com/{man['repo']}/releases/download/{man['tag']}/"
    record = {"model_id": model_id, "prebuilt": True, "release": man["tag"], "results": {}, "commands": []}
    for a in ent["assets"]:
        p = os.path.join(out_dir, a["file"])
        got = download(base + a["asset"], p, a["sha256"])
        record["results"][a["file"]] = {"file_sha256": got, "data_sha256": data_sha256(p), "identical_data": True}
    for extra in ("licence", "attribution"):
        if ent.get(extra):
            record[extra] = ent[extra]
    with open(os.path.join(out_dir, "fetch_record.json"), "w") as f:
        json.dump(record, f, indent=1)
    log(f"{model_id}: {len(ent['assets'])} file(s) from release {man['tag']}; weights licence: {ent.get('licence')}")
    return record


def fetch(model_id: str, *, eval_dir: str | None = None, files: dict | None = None, sources: dict | None = None,
          python: dict | None = None, int8: bool = False, calib_dir: str | None = None, synthetic_n: int = 8,
          dry_run: bool = False, threads: int = 4) -> dict:
    rec = zoo.recipes().get(model_id)
    if not rec:
        raise FetchError(f"no recipe for {model_id!r} (see `bitino list --converted`)")
    files, sources, python = files or {}, sources or {}, python or {}
    home = zoo.home()
    dl_dir = os.path.join(home, "downloads", model_id)
    out_dir = zoo.model_dir(model_id)
    work = os.path.join(home, "work", model_id)
    log(f"{model_id}: recipe state {rec['state']}, env {rec['env']}, converter {rec['converter']}")
    for n in rec.get("notes", []):
        log(f"note: {n}")
    if rec["state"] == "broken":
        b = rec.get("broken") or {}
        if not (b.get("download") and b["download"] in (files or {})):
            raise FetchError(f"{model_id}: this recipe cannot run today: {b.get('reason', 'see registry/recipes_verified.json')}"
                             + (f"\n{b['instructions']}" if b.get("instructions") else ""))
    if rec["state"] == "needs-manual-step" and (rec.get("manual_step") or {}).get("instructions"):
        log(f"manual step: {rec['manual_step']['instructions']}")
    ctx = {"model_id": model_id, "downloads": dl_dir, "out_dir": out_dir, "work": work, "threads": str(max(1, threads))}
    record = {"model_id": model_id, "recipe_state": rec["state"], "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "downloads": [], "sources": [], "commands": []}
    if dry_run:
        print(json.dumps(rec, indent=1))
        return record
    if not zoo.source_checkout():
        raise FetchError(f"converting {model_id} from its original source runs the converters of a source checkout "
                         f"(tools/) in the reference environment of the recipe ({rec['env']}); they are not part of the "
                         f"installed package. Clone {zoo.REPO_URL}, `pip install -e .` there and run "
                         f"`bitino fetch {model_id}`: the converted file lands in {out_dir}, where "
                         f"bitino.load('{model_id}') finds it.")
    new_dl = not os.path.isdir(dl_dir)
    os.makedirs(dl_dir, exist_ok=True)
    if new_dl:
        log(f"new download directory {dl_dir}")
    for i, d in enumerate(rec.get("downloads", [])):
        name = d.get("name", f"file{i}")
        if name in files:
            path = os.path.abspath(os.path.expanduser(files[name]))
            got = sha256_file(path)
            if d.get("member") and d.get("archive_sha256") == got:   # the whole zip release was passed
                path = os.path.join(dl_dir, d["file"])
                got = extract_member(os.path.abspath(os.path.expanduser(files[name])), d["member"], path, d.get("sha256"))
            if d.get("sha256") and got != d["sha256"]:
                raise FetchError(f"{path}: SHA-256 {got} does not match the recorded {d['sha256']}")
        else:
            if d.get("manual") and not d.get("url"):
                raise FetchError(f"{name}: {d['manual']}")
            if d.get("member"):   # one file of a zip release: download the archive, extract the member
                arch = os.path.join(dl_dir, d["archive_file"])
                try:
                    download(d["url"], arch, d.get("archive_sha256"))
                except FetchError as e:
                    raise FetchError(f"{e}" + (f"\n{d['manual']}" if d.get("manual") else ""))
                path = os.path.join(dl_dir, d["file"])
                got = extract_member(arch, d["member"], path, d.get("sha256"))
            else:
                path = os.path.join(dl_dir, d["file"])
                try:
                    got = download(d["url"], path, d.get("sha256"))
                except FetchError as e:
                    raise FetchError(f"{e}" + (f"\n{d['manual']}" if d.get("manual") else ""))
                except Exception as e:
                    raise FetchError(f"download of {d['url']} failed: {e}" + (f"\n{d['manual']}" if d.get("manual") else ""))
        ctx["download:" + name] = path
        if i == 0:
            ctx["weights"] = path
            ctx["sha256"] = got
        record["downloads"].append({"name": name, "url": d.get("url"), "sha256": got, "verified": bool(d.get("sha256"))})
    for s in rec.get("sources", []):
        name = s.get("name", "repo")
        path = os.path.abspath(os.path.expanduser(sources[name])) if name in sources else get_source(s, home)
        ctx[name] = path
        record["sources"].append({"name": name, "path": path, **{k: s.get(k) for k in ("repo", "commit", "package", "version")}})
    for name, path in sources.items():
        ctx.setdefault(name, os.path.abspath(os.path.expanduser(path)))
    ctx["eval_dir"] = os.path.abspath(eval_dir) if eval_dir else synthetic_eval(rec, home, synthetic_n)
    if not eval_dir and not rec["eval"].get("synthetic_ok", True):
        log("WARNING: this model's converter expects a real eval set (text, image-to-image or embedding targets); "
            "pass --eval-dir if the synthetic one is rejected")
    if calib_dir:
        ctx["calib_dir"] = os.path.abspath(calib_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1")
    for step in rec["steps"]:
        py = resolve_python(step["env"], python)
        # {python:ENV} placeholders (a second reference environment the converter calls, e.g. Python 2.7 Theano)
        ctx_py = {f"python:{k}": resolve_python(k, python)
                  for a in step["args"] for k in re.findall(r"\{python:([\w-]+)\}", a)}
        argv = [py, "-I", os.path.join(zoo.ROOT, step["tool"])] + [_fmt(a, ctx | ctx_py) for a in step["args"]]
        log("run " + " ".join(argv))
        record["commands"].append(argv)
        r = subprocess.run(argv, cwd=work, env=env)
        if r.returncode != 0:
            raise FetchError(f"step {step['tool']} failed with exit code {r.returncode}")
    record["results"] = {}
    int8_files = {q["file"] for q in rec.get("int8", [])}
    for fn, exp in rec.get("expected", {}).items():
        if fn in int8_files:
            continue   # checked after quantization (--int8)
        p = os.path.join(out_dir, fn)
        if not os.path.exists(p):
            continue
        got = data_sha256(p)
        same = got == exp.get("data_sha256")
        gsame = graph_sha256(p) == exp.get("graph_sha256") if exp.get("graph_sha256") else None
        record["results"][fn] = {"data_sha256": got, "expected": exp.get("data_sha256"), "identical_data": same,
                                 "identical_graph": gsame, "file_sha256": sha256_file(p)}
        log(f"{fn}: data section {'IDENTICAL to' if same else 'DIFFERS from'} the published conversion ({got[:16]}); "
            f"graph {'identical' if gsame else 'differs' if gsame is False else 'not recorded'}")
    if int8:
        if not calib_dir:
            raise FetchError("--int8 needs --calib-dir (calibration images from the training split; see docs/int8.md)")
        for q in rec.get("int8", []):
            argv = [sys.executable, "-I", os.path.join(zoo.ROOT, "tools", "quantize_int8_full.py")] + [_fmt(a, ctx) for a in q["args"]]
            log("run " + " ".join(argv))
            record["commands"].append(argv)
            r = subprocess.run(argv, cwd=work, env=env)
            if r.returncode != 0:
                raise FetchError(f"quantization of {q['file']} failed with exit code {r.returncode}")
            p = os.path.join(out_dir, q["file"])
            got = data_sha256(p)
            exp = rec.get("expected", {}).get(q["file"], {}).get("data_sha256")
            gx = rec.get("expected", {}).get(q["file"], {}).get("graph_sha256")
            record["results"][q["file"]] = {"data_sha256": got, "expected": exp, "identical_data": got == exp,
                                            "identical_graph": (graph_sha256(p) == gx) if gx else None}
            g = record['results'][q['file']]['identical_graph']
            log(f"{q['file']}: data section {'IDENTICAL to' if got == exp else 'DIFFERS from'} the published file; "
                f"graph {'identical' if g else 'differs' if g is False else 'not recorded'}")
    record["bytes_downloaded"] = BYTES_DOWNLOADED[0]
    with open(os.path.join(out_dir, "fetch_record.json"), "w") as f:
        json.dump(record, f, indent=1)
    return record
