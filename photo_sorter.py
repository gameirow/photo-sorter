"""
PHOTO/SORTER — sorts a folder of photos into category folders by what's in them.
Local web app: Python backend (SigLIP 2 zero-shot, GPU if available) + ui.html frontend.

Run:  python photo_sorter.py   -> opens http://127.0.0.1:7860
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    # Launcher: open the window on the intro animation at once, then load the heavy engine (PyTorch etc.) behind it.
    _here = Path(__file__).resolve().parent
    if sys.stdout is None or sys.stderr is None:  # launched windowless (pythonw): log to a file
        _log = open(_here / "sorter.log", "a", buffering=1, encoding="utf-8")
        sys.stdout, sys.stderr = sys.stdout or _log, sys.stderr or _log
    sys.path.insert(0, str(_here))
    try:
        import webview
    except ImportError:  # no pywebview: plain browser, the intro plays inside the page
        import threading
        import webbrowser
        import photo_sorter as engine
        url = engine.start_server()
        webbrowser.open(url + "/app")
        print(f"Sorter running at {url}  (Ctrl+C to stop)")
        threading.Event().wait()
    else:
        try:  # own taskbar identity: Windows shows Sorter's icon instead of grouping it under Python
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sorter.PhotoSorter")
        except Exception:  # noqa: BLE001
            pass
        import json as _json
        import os as _os
        import tempfile as _tempfile
        try:
            _sound = _json.loads((Path(_tempfile.gettempdir()) / "photo_sorter_thumbs" / "settings.json").read_text()).get("sound", True)
        except Exception:  # noqa: BLE001
            _sound = True
        # let the intro play its sound without a click first (WebView2 reads extra browser flags from this variable)
        _os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = (
            _os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "") +
            " --disable-features=ElasticOverscroll --autoplay-policy=no-user-gesture-required").strip()
        _intro = ("<!doctype html><html><head><meta charset='utf-8'></head><body style='margin:0;background:#0B0A0E'>"
                  f"<script>window.SPLASH_MODE='intro';window.SPLASH_SOUND={'true' if _sound else 'false'}</script>" + (_here / "splash.html").read_text(encoding="utf-8") + "</body></html>")
        _win = webview.create_window("Sorter", html=_intro, width=1480, height=920, min_size=(1180, 720),
                                     background_color="#0B0A0E")

        def _boot():
            import json
            import threading
            import traceback
            try:
                import photo_sorter as engine
                engine.WIN["w"] = _win
                url = engine.start_server()
                _win.events.loaded += lambda *a: engine.bind_drop(_win, url)
                threading.Thread(target=engine.preload, daemon=True).start()
                _win.events.loaded.wait(15)
                _win.evaluate_js(f"splashGo({json.dumps(url + '/?from=splash')})")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                _win.evaluate_js(f"splashError({json.dumps(f'Sorter could not start: {e}')})")

        webview.start(_boot, icon=str(_here / "sorter.ico"))
    sys.exit(0)

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

try:  # iPhone .HEIC support
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
if sys.stdout is None or sys.stderr is None:  # launched windowless (pythonw): log to a file
    _log = open(HERE / "sorter.log", "a", buffering=1, encoding="utf-8")
    sys.stdout, sys.stderr = sys.stdout or _log, sys.stderr or _log
WIN = {"w": None}  # native window, when pywebview is available
MODELS = {
    "accurate": "google/siglip2-so400m-patch14-384",
    "fast": "google/siglip2-base-patch16-224",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
RAW_EXTS = {".raf", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2"}
SIDE_EXTS = {".aae", ".xmp"}  # edit sidecars (iPhone .AAE, Lightroom .XMP): follow their photo/video
VID_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".webm", ".3gp"}
UNSORTED = "_unsorted"
AROLL, BROLL, SEP = "A-roll", "B-roll", " · "
TALK_PROMPT = "a person talking to the camera, vlog selfie video"
N_FRAMES = 8  # frames per video: averaged for sorting, kept one by one for search (find the moment)
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 1))
BATCH = 16
OUT_NAME = "_sorted"
THUMBS = Path(tempfile.gettempdir()) / "photo_sorter_thumbs"
THUMBS.mkdir(exist_ok=True)

DEFAULT_CATS = """portrait: a portrait photo of a person
selfie: a selfie taken with a phone's front camera
group: a photo of a group of people
party & events: a photo of a party, a concert or a celebration
animals: a photo of an animal or a pet
food: a photo of food, a meal or a drink
landscape: a landscape photo of nature or mountains
snow & mountains: a photo of snow, skiing or snowy mountains
travel & landmarks: a photo of a famous landmark or a tourist spot
beach & sea: a photo of a beach, the sea or a lake
city & architecture: a photo of a city street or buildings
plants & flowers: a close-up photo of plants or flowers
sport & fitness: a photo of sport, the gym or working out
outfits & fashion: a photo of an outfit or streetwear
cars & vehicles: a photo of a car, a motorbike or another vehicle
night: a photo taken at night
screenshots & documents: a screenshot or a photo of a document"""


# ---------------------------------------------------------------- model
class Classifier:
    def __init__(self, model_id):
        from transformers import AutoModel, AutoProcessor  # lazy: keeps app start instant
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.label = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"

    @staticmethod
    def _norm(x):
        if not torch.is_tensor(x):  # newer transformers may return an output object
            x = x.pooler_output if getattr(x, "pooler_output", None) is not None else x[0]
        return x / x.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text(self, texts):
        t = self.proc(text=texts, padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt").to(self.device)
        return self._norm(self.model.get_text_features(**t)).float().cpu()

    @torch.no_grad()
    def images(self, ims):
        x = self.proc(images=ims, return_tensors="pt").to(self.device)
        x["pixel_values"] = x["pixel_values"].to(self.dtype)
        return self._norm(self.model.get_image_features(**x)).float().cpu()

    @torch.no_grad()
    def logits(self, img_e, txt_e):
        scale = self.model.logit_scale.exp().float().cpu()
        bias = self.model.logit_bias.float().cpu()
        return img_e @ txt_e.T * scale + bias

    def scores(self, img_e, txt_e):
        return torch.sigmoid(self.logits(img_e, txt_e))


_clf, _clf_lock = {}, threading.Lock()
def get_clf(key):
    with _clf_lock:
        if key not in _clf:
            _clf.clear()
            _clf[key] = Classifier(MODELS[key])
        return _clf[key]


# ---------------------------------------------------------------- helpers
def parse_cats(text):
    names, prompts = [], []
    entries = []
    for line in text.splitlines():  # commas only separate entries on lines without a description
        entries += [line] if ":" in line else line.split(",")
    for line in entries:
        line = line.strip()
        if not line:
            continue
        name, _, prompt = line.partition(":")
        name = name.strip()
        prompt = prompt.strip() or f"a photo of {name}"
        if name and name not in names:
            names.append(name)
            prompts.append(prompt)
    if len(names) < 2:
        raise HTTPException(400, "Add at least 2 categories.")
    return names, prompts


def safe(name):
    return re.sub(r'[<>:"/\\|?*]', "-", name).strip(" .") or "category"


def open_image(p, size):
    if p.suffix.lower() in RAW_EXTS:  # use the JPEG preview embedded in the raw file
        import rawpy
        with rawpy.imread(str(p)) as r:
            t = r.extract_thumb()
        im = Image.open(io.BytesIO(t.data)) if t.format == rawpy.ThumbFormat.JPEG else Image.fromarray(t.data)
    else:
        im = Image.open(p)
        if im.format == "JPEG":
            im.draft("RGB", (size * 2, size * 2))  # fast reduced decode
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    return im


_vad, _vad_lock = {}, threading.Lock()
def speech_ratio(p, max_s=120):
    """Share of the clip's audio that is human speech (Silero VAD)."""
    import av
    import numpy as np
    with av.open(str(p)) as c:
        if not c.streams.audio:
            return 0.0
        rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks, n = [], 0
        for fr in c.decode(c.streams.audio[0]):
            for o in rs.resample(fr):
                a = o.to_ndarray().reshape(-1)
                chunks.append(a)
                n += a.size
            if n >= max_s * 16000:
                break
    if not chunks:
        return 0.0
    from silero_vad import get_speech_timestamps, load_silero_vad
    wav = torch.from_numpy(np.concatenate(chunks)[: max_s * 16000].astype(np.float32) / 32768)
    with _vad_lock:  # the VAD model keeps internal state: one clip at a time
        if "m" not in _vad:
            _vad["m"] = load_silero_vad()
        ts = get_speech_timestamps(wav, _vad["m"], sampling_rate=16000)
    return round(sum(t["end"] - t["start"] for t in ts) / max(1, wav.numel()), 3)


def video_frames(p, n=N_FRAMES):
    import av
    out = []
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        try:
            s.codec_context.skip_frame = "NONKEY"  # jump straight to keyframes: much faster seeking
        except Exception:  # noqa: BLE001
            pass
        dur = c.duration / 1e6 if c.duration else float((s.duration or 0) * (s.time_base or 0))
        for k in range(n):
            try:
                c.seek(int(dur * (k + 0.5) / n * 1e6), any_frame=False, backward=True)
            except Exception:  # noqa: BLE001
                pass
            for f in c.decode(s):
                im = f.to_image()
                rot = getattr(f, "rotation", 0) or 0  # phone clips store rotation as metadata
                if rot:
                    im = im.rotate(rot, expand=True)
                im.thumbnail((384, 384), Image.LANCZOS)
                out.append(im.convert("RGB"))
                break
    if not out:
        raise RuntimeError("no decodable frames")
    return out, dur


def prepare(p, is_video):
    """Thumbnails (+ frames and speech for videos), cached across runs."""
    key = hashlib.md5(f"{p}|{p.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    small, large, meta = THUMBS / f"{key}_s.jpg", THUMBS / f"{key}_l.jpg", THUMBS / f"{key}_v.json"
    info = {"key": key, "s": f"/thumbs/{small.name}", "l": f"/thumbs/{large.name}", "kind": "photo",
            "frames": [large.name], "speech": 0.0, "dur": 0.0}
    if not is_video:
        if not (small.exists() and large.exists()):
            im = open_image(p, 384)
            im.save(large, quality=85)
            ImageOps.fit(im, (96, 96)).save(small, quality=80)
        return info
    if meta.exists():
        info.update(json.loads(meta.read_text()))
        return info
    frames, dur = video_frames(p)
    names = []
    for k, im in enumerate(frames):
        im.save(THUMBS / f"{key}_f{k}.jpg", quality=85)
        names.append(f"{key}_f{k}.jpg")
    mid = frames[len(frames) // 2]
    mid.save(large, quality=85)
    ImageOps.fit(mid, (96, 96)).save(small, quality=80)
    try:
        sp = speech_ratio(p)
    except Exception as e:  # noqa: BLE001
        print(f"audio skipped for {p.name}: {e}")
        sp = 0.0
    extra = {"kind": "video", "frames": names, "speech": sp, "dur": round(dur, 1)}
    meta.write_text(json.dumps(extra))
    info.update(extra)
    return info


def scan(root, recursive, videos=True):
    files = root.rglob("*") if recursive else root.iterdir()
    imgs, raws, sides = [], {}, {}
    for p in files:
        if not p.is_file() or OUT_NAME in p.relative_to(root).parts:
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS or (videos and ext in VID_EXTS):
            imgs.append(p)
        elif ext in RAW_EXTS:
            raws.setdefault((p.parent, p.stem.lower()), []).append(p)
        elif ext in SIDE_EXTS:
            sides.setdefault((p.parent, p.stem.lower()), []).append(p)
    keys = {(p.parent, p.stem.lower()) for p in imgs}
    try:
        import rawpy  # noqa: F401  raw-only shots get classified from their preview
        raw_only = [r[0] for k, r in raws.items() if k not in keys]
    except ImportError:
        raw_only = []
    return sorted(imgs) + sorted(raw_only), raws, sides


def assign(items, scores, names, threshold):
    """scores: [N, len(names)+1]; the last column is the talking-to-camera prompt."""
    conf = torch.logit(scores[:, :len(names)].clamp(1e-7, 1 - 1e-7)).softmax(dim=1)  # share of belief, sums to 1
    best = conf.max(dim=1)
    out = []
    for it, s, i, tk in zip(items, best.values.tolist(), best.indices.tolist(), scores[:, -1].tolist()):
        c = names[i] if s >= threshold else UNSORTED
        if it.get("kind") == "video":
            sp = it.get("speech", 0.0)
            if sp >= 0.3 and (tk >= 0.01 or sp >= 0.6):  # mostly speech + a person facing camera
                c, s = AROLL, max(s, tk)
            else:
                c = f"{BROLL}{SEP}{c}"
        out.append((c, round(float(s), 3)))
    return out


def embed_items(clf, chunk):
    """One embedding per item (videos average their frames) + the per-frame embeddings."""
    ims, owner = [], []
    for j, it in enumerate(chunk):
        for f in it["frames"]:
            ims.append(Image.open(THUMBS / f).convert("RGB"))
            owner.append(j)
    e = clf.images(ims)
    owner = torch.tensor(owner)
    frames = [e[owner == j] for j in range(len(chunk))]
    out = torch.stack([f.mean(0) for f in frames])
    return out / out.norm(dim=-1, keepdim=True), frames


def valid_cat(c):
    base = S["names"] + [UNSORTED]
    return c in base or c == AROLL or (c.startswith(BROLL + SEP) and c[len(BROLL + SEP):] in base)


def unique(dst):
    n, base = 1, dst
    while dst.exists():
        dst = base.with_name(f"{base.stem}_{n}{base.suffix}")
        n += 1
    return dst


class EmbCache:
    """Image embeddings on disk, per model: re-analyzing a folder skips the GPU entirely."""
    def __init__(self):
        self.model, self.d, self.dirty = None, {}, 0

    def use(self, model):
        if model != self.model:
            self.save()
            self.model = model
            try:
                self.d = torch.load(THUMBS / f"emb_{model}.pt")
            except Exception:  # noqa: BLE001
                self.d = {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v.half()
        self.dirty += 1
        if self.dirty >= 500:
            self.save()

    def save(self):
        if self.model and self.dirty:
            tmp = THUMBS / f"emb_{self.model}.tmp"
            torch.save(self.d, tmp)
            os.replace(tmp, THUMBS / f"emb_{self.model}.pt")
            self.dirty = 0


EMBC = EmbCache()
SESS = THUMBS / "sessions"
SESS.mkdir(exist_ok=True)


def session_file(root):
    return SESS / (hashlib.md5(str(Path(root).resolve()).lower().encode()).hexdigest()[:16] + ".json")


def load_overrides(root):
    try:
        return json.loads(session_file(root).read_text()).get("overrides", {})
    except Exception:  # noqa: BLE001
        return {}


def save_overrides():
    if S["root"]:
        session_file(S["root"]).write_text(json.dumps({"root": S["root"], "overrides": S["over"]}))


# ---------------------------------------------------------------- job state
LOCK = threading.Lock()
S = {"phase": "idle", "msg": "READY", "run": 0, "total": 0, "done": 0, "items": [], "log": [],
     "names": [], "prompts": [], "threshold": 0.0, "model": "accurate", "device": "",
     "root": "", "over": {}}


def set_cat(it, c, s):
    """Apply a result, honouring the user's saved manual fixes; caller holds LOCK."""
    ov = S["over"].get(it["path"])
    if ov and valid_cat(ov):
        c, s = ov, 1.0
    it["cat"], it["score"] = c, s
    S["log"].append(it["id"])
EMB = {"img": None, "frames": None, "fidx": []}  # fidx[j] = (item id, frame k, frames n)
RUNNING = ("scanning", "loading", "sorting")


def prepare_safe(p):
    try:
        return prepare(p, p.suffix.lower() in VID_EXTS)
    except Exception as e:  # noqa: BLE001
        print(f"skip {p.name}: {e}")
        return None


def job(root, recursive, videos):
    """Decode in parallel threads while the GPU classifies whatever is ready."""
    embd, holder, pending = {}, {}, []
    state = {"txt": None}

    def load():
        try:
            holder["clf"] = get_clf(S["model"])
        except Exception as e:  # noqa: BLE001
            holder["err"] = e

    def flush(force=False):
        if "err" in holder:
            raise holder["err"]
        if "clf" not in holder:
            return
        clf = holder["clf"]
        if state["txt"] is None:
            state["txt"] = clf.text(S["prompts"] + [TALK_PROMPT])
            S["device"], S["phase"] = clf.label, "sorting"
        while pending and (force or len(pending) >= BATCH):
            chunk = pending[:BATCH]
            del pending[:BATCH]
            todo = [it for it in chunk if EMBC.get(it["key"]) is None
                    or (it["kind"] == "video" and EMBC.get(it["key"] + "#f") is None)]
            if todo:
                embs, frames = embed_items(clf, todo)
                for it, e, fr in zip(todo, embs, frames):
                    EMBC.put(it["key"], e)
                    if it["kind"] == "video":
                        EMBC.put(it["key"] + "#f", fr)
            e = torch.stack([EMBC.get(it["key"]).float() for it in chunk])
            res = assign(chunk, clf.scores(e, state["txt"]), S["names"], S["threshold"])
            with LOCK:
                for it, emb, (c, sc) in zip(chunk, e, res):
                    embd[it["id"]] = emb
                    set_cat(it, c, sc)
                S["done"] += len(chunk)

    try:
        files, raws, sides = scan(root, recursive, videos)
        if not files:
            raise RuntimeError("No photos or videos found in that folder.")
        S["total"] = len(files)
        EMBC.use(S["model"])
        threading.Thread(target=load, daemon=True).start()  # model loads while files decode

        with ThreadPoolExecutor(WORKERS) as ex:
            for p, info in zip(files, ex.map(prepare_safe, files)):
                if info is None:
                    S["total"] -= 1
                    continue
                k = (p.parent, p.stem.lower())
                extras = ([] if p.suffix.lower() in RAW_EXTS | VID_EXTS else raws.get(k, [])) + sides.pop(k, [])
                with LOCK:
                    it = {"id": len(S["items"]), "name": p.name, "path": str(p),
                          "extras": [str(x) for x in extras], **info, "cat": None, "score": 0.0}
                    S["items"].append(it)
                pending.append(it)
                flush()

        if "clf" not in holder:
            S["phase"], S["msg"] = "loading", "LOADING MODEL — FIRST RUN DOWNLOADS IT"
            while "clf" not in holder and "err" not in holder:
                threading.Event().wait(0.1)
        flush(force=True)
        EMBC.save()
        EMB["img"] = torch.stack([embd[i] for i in range(len(S["items"]))])
        build_search_index()
        S["phase"], S["msg"] = "done", f"COMPLETE — {len(S['items'])} FILES CLASSIFIED"
    except Exception as e:  # noqa: BLE001
        EMBC.save()
        S["phase"], S["msg"] = "error", f"ERROR: {e}"


# ---------------------------------------------------------------- API
app = FastAPI()
app.mount("/thumbs", StaticFiles(directory=THUMBS), name="thumbs")


class StartReq(BaseModel):
    folder: str
    cats: str
    model: str = "accurate"
    recursive: bool = True
    videos: bool = True
    threshold: float = 0.0

class ResortReq(BaseModel):
    cats: str
    threshold: float = 0.0

class MoveReq(BaseModel):
    ids: list[int]
    cat: str

class OrgReq(BaseModel):
    mode: str = "Copy"
    out: str

class PathReq(BaseModel):
    path: str


@app.get("/")
def index():
    return HTMLResponse(UI_PAGE("outro"))


@app.get("/app")
def index_full():  # browser fallback: plays the whole intro inside the page
    return HTMLResponse(UI_PAGE("full"))


def UI_PAGE(mode):
    ui = (HERE / "ui.html").read_text(encoding="utf-8")
    intro = f"<script>window.SPLASH_MODE='{mode}';window.SPLASH_SOUND={'true' if settings().get('sound', True) else 'false'}</script>" + (HERE / "splash.html").read_text(encoding="utf-8")
    return ui.replace("<body>", "<body>" + intro, 1)


@app.get("/api/defaults")
def defaults():
    return {"cats": DEFAULT_CATS}


@app.get("/api/pick")
def pick():
    """Native folder picker on this machine."""
    if WIN["w"] is not None:
        import webview
        kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
        res = WIN["w"].create_file_dialog(kind)
        if not res:
            return {"path": ""}
        return {"path": res[0] if isinstance(res, (list, tuple)) else res}
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Choose a photo folder")
        root.destroy()
        return {"path": path or ""}
    except Exception as e:  # noqa: BLE001
        return {"path": "", "error": str(e)}


@app.post("/api/start")
def start(r: StartReq):
    if S["phase"] in RUNNING:
        raise HTTPException(409, "Already running.")
    root = Path(r.folder.strip().strip('"'))
    if not r.folder.strip() or not root.is_dir():
        raise HTTPException(400, f"Folder not found: {root}")
    if r.model not in MODELS:
        raise HTTPException(400, "Unknown model.")
    names, prompts = parse_cats(r.cats)
    with LOCK:
        S.update(phase="scanning", msg="SCANNING", run=S["run"] + 1, total=0, done=0, items=[], log=[],
                 names=names, prompts=prompts, threshold=r.threshold, model=r.model, device="",
                 root=str(root), over=load_overrides(root))
    EMB["img"], EMB["frames"], EMB["fidx"] = None, None, []
    threading.Thread(target=job, args=(root, r.recursive, r.videos), daemon=True).start()
    return {"run": S["run"], "out": str(root)}  # default: organise the chosen folder in place


@app.get("/api/state")
def state(have: int = 0, run: int = -1, since: int = 0):
    with LOCK:
        items, log = S["items"], S["log"]
        same = run == S["run"]
        start_at, since = (have, since) if same else (0, 0)
        return {"phase": S["phase"], "msg": S["msg"], "run": S["run"], "total": S["total"],
                "done": S["done"], "names": S["names"], "device": S["device"],
                "new": [{**{k: it[k] for k in ("id", "name", "s", "l", "dur", "speech")}, "v": it["kind"] == "video"}
                        for it in items[start_at:]],
                "changes": [[i, items[i]["cat"], items[i]["score"]] for i in log[since:]],
                "since": len(log), "fixes": len(S["over"])}


@app.post("/api/resort")
def resort(r: ResortReq):
    if S["phase"] != "done" or EMB["img"] is None:
        raise HTTPException(409, "Analyze a folder first.")
    names, prompts = parse_cats(r.cats)
    clf = get_clf(S["model"])
    res = assign(S["items"], clf.scores(EMB["img"], clf.text(prompts + [TALK_PROMPT])), names, r.threshold)
    with LOCK:
        S.update(names=names, prompts=prompts, threshold=r.threshold)
        for it, (c, s) in zip(S["items"], res):
            set_cat(it, c, s)
        cats = [[it["cat"], it["score"]] for it in S["items"]]
    return {"names": names, "cats": cats}


@app.post("/api/move")
def move(r: MoveReq):
    if not valid_cat(r.cat):
        raise HTTPException(400, "Unknown category.")
    with LOCK:
        for i in r.ids:
            if 0 <= i < len(S["items"]):
                it = S["items"][i]
                S["over"][it["path"]] = r.cat  # remembered for this folder, across restarts
                it["cat"], it["score"] = r.cat, 1.0
        save_overrides()
    return {"ok": True}


@app.post("/api/organize")
def organize(r: OrgReq):
    items = S["items"]
    if not items or S["phase"] != "done":
        raise HTTPException(409, "Analyze a folder first.")
    out = Path(r.out.strip().strip('"'))
    log, errors, emptied = [], [], set()
    log_file = out / "_sort_log.json"
    try:
        prev = json.loads(log_file.read_text()) if log_file.exists() else None
    except Exception:  # noqa: BLE001
        prev = None
    older = prev["files"] if prev and prev.get("mode") == "Move" else []

    def write_log():  # atomic: a crash mid-write can't corrupt the record Undo depends on
        if r.mode == "Move" and (older or log):
            out.mkdir(parents=True, exist_ok=True)
            tmp = log_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"mode": "Move", "files": older + log}, indent=1))
            os.replace(tmp, log_file)

    try:
        _organize(items, out, r.mode, log, errors, emptied, write_log)
    finally:  # always record what moved, even if something failed halfway, so Undo can put it back
        if r.mode == "Move":
            write_log()
        elif log and not (prev and prev.get("mode") == "Move"):  # never overwrite a log Undo still needs
            log_file.write_text(json.dumps({"mode": r.mode, "files": log}, indent=1))
        save_overrides()
    if r.mode == "Move":  # remove only the folders this move emptied (Undo recreates them)
        root = Path(S["root"]).resolve()
        for d in sorted(emptied, key=lambda p: len(p.parts), reverse=True):
            while d != root and root in d.parents and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
    verb = "MOVED" if r.mode == "Move" else "COPIED"
    note = f" · {len(errors)} COULD NOT BE MOVED (SEE SORTER.LOG)" if errors else ""
    for e in errors:
        print("organize:", e)
    return {"msg": f"{verb} {len(log)} FILES{note} → {out}", "errors": len(errors)}


def _organize(items, out, mode, log, errors, emptied, checkpoint=lambda: None):
    for it in items:
        dest = out.joinpath(*[safe(x) for x in it["cat"].split(SEP)])  # "B-roll · city" -> B-roll/city
        dest.mkdir(parents=True, exist_ok=True)
        moved = []
        for f in [it["path"], *it["extras"]]:
            src = Path(f)
            if not src.exists() or (mode == "Move" and src.parent.resolve() == dest.resolve()):
                moved.append(f)  # missing, or already in its category folder
                continue
            dst = unique(dest / src.name)
            try:
                (shutil.move if mode == "Move" else shutil.copy2)(str(src), str(dst))
            except OSError as e:  # locked by OneDrive / another app: skip it, keep going
                errors.append(f"{src}: {e}")
                if mode == "Move" and src.exists() and dst.exists():  # a move falls back to copy+delete:
                    try:                                             # drop the half-done copy, the original is intact
                        dst.unlink()
                    except OSError:
                        pass
                moved.append(f)
                continue
            log.append([str(src), str(dst)])
            moved.append(str(dst))
            emptied.add(src.parent.resolve())
            if len(log) % 20 == 0:  # save progress as we go: even a crash or power cut leaves Undo a record
                checkpoint()
        if mode == "Move":  # keep paths (and saved fixes) current so re-organizing works
            if it["path"] in S["over"]:
                S["over"][moved[0]] = S["over"].pop(it["path"])
            it["path"], it["extras"] = moved[0], moved[1:]


@app.post("/api/undo")
def undo(r: PathReq):
    log_file = Path(r.path.strip().strip('"')) / "_sort_log.json"
    if not log_file.exists():
        raise HTTPException(404, "No sort log in that output folder.")
    data = json.loads(log_file.read_text())
    if data["mode"] != "Move":
        return {"msg": "THAT RUN MADE COPIES — ORIGINALS UNTOUCHED. DELETE THE OUTPUT FOLDER IF UNWANTED."}
    back, n = {}, 0
    for src, dst in reversed(data["files"]):
        if Path(dst).exists():
            Path(src).parent.mkdir(parents=True, exist_ok=True)
            final = unique(Path(src))
            shutil.move(dst, final)
            back[dst] = str(final)
            n += 1
    def orig(p, hops=0):  # follow a file back through several organise runs
        while p in back and hops < 50:
            p, hops = back[p], hops + 1
        return p
    S["over"] = {orig(k): v for k, v in S["over"].items()}
    save_overrides()
    for it in S["items"]:
        it["path"] = orig(it["path"])
        it["extras"] = [orig(x) for x in it["extras"]]
    log_file.unlink()
    for d in {Path(dst).parent for _, dst in data["files"]}:  # remove category folders the undo emptied
        while d != log_file.parent and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    return {"msg": f"RESTORED {n} FILES TO THEIR ORIGINAL PLACE"}


@app.post("/api/open")
def open_folder(r: PathReq):
    p = Path(r.path.strip().strip('"'))
    if not p.exists():
        raise HTTPException(404, "Folder doesn't exist yet.")
    if sys.platform.startswith("win"):
        os.startfile(p)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])
    return {"ok": True}


def start_server():
    import socket
    import time
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------- search
def build_search_index():
    """Every photo, and every sampled moment of every video, as one row: search ranks them all."""
    rows, fidx = [], []
    for it in S["items"]:
        fr = EMBC.get(it["key"] + "#f") if it["kind"] == "video" else None
        if fr is not None:
            rows.append(fr.float())
            fidx += [(it["id"], k, len(fr)) for k in range(len(fr))]
        else:
            rows.append(EMB["img"][it["id"]][None])
            fidx.append((it["id"], 0, 1))
    EMB["frames"], EMB["fidx"] = torch.cat(rows), fidx


@app.get("/api/search")
def search(q: str, limit: int = 200):
    """Find photos and video moments matching free text, anywhere in the analyzed folder. Moves nothing."""
    q = q.strip()
    if not q:
        raise HTTPException(400, "Type something to search for.")
    if S["phase"] != "done" or EMB["frames"] is None:
        raise HTTPException(409, "Analyze a folder first — search looks inside the analyzed folder.")
    clf = get_clf(S["model"])
    # match strength per photo / video moment; two phrasings, keep the stronger (each wins on different photos)
    l = clf.logits(EMB["frames"], clf.text([q, f"a photo of {q}"])).max(dim=1).values.tolist()
    best = {}
    for j, (i, k, n) in enumerate(EMB["fidx"]):
        if i not in best or l[j] > best[i][0]:
            best[i] = (l[j], k, n)
    ranked = sorted(best.items(), key=lambda x: -x[1][0])
    out = []
    for i, (sc, k, n) in ranked[:limit]:
        if sc < SEARCH_MAYBE:
            break
        cut = SEARCH_MATCH
        it = S["items"][i]
        video = it["kind"] == "video"
        out.append({"id": i, "name": it["name"], "path": it["path"], "cat": it["cat"], "score": round(sc, 2),
                    "strong": sc >= cut, "video": video,
                    "thumb": f"/thumbs/{it['frames'][k]}" if video and k < len(it["frames"]) else it["l"],
                    "at": round(it["dur"] * (k + .5) / n, 1) if video else None})
    return {"q": q, "results": out, "searched": len(best)}


# SigLIP logits, measured on real photos: true matches scored -3..-7.5, absent things stayed below -10
SEARCH_MATCH, SEARCH_MAYBE = -8.5, -10.0


@app.post("/api/reveal")
def reveal(r: PathReq):
    """Show a file selected in File Explorer."""
    p = Path(r.path)
    if not p.exists():
        raise HTTPException(404, "That file isn't there any more.")
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", str(p)])
    else:
        open_folder(PathReq(path=str(p.parent)))
    return {"ok": True}


SETTINGS = THUMBS / "settings.json"


def settings():
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:  # noqa: BLE001
        return {}


class SettingsReq(BaseModel):
    sound: bool


@app.get("/api/settings")
def get_settings():
    return {"sound": settings().get("sound", True)}


@app.post("/api/settings")
def set_settings(r: SettingsReq):
    SETTINGS.write_text(json.dumps({**settings(), "sound": r.sound}))
    return {"sound": r.sound}


def preload():
    """Warm the default model up while the user picks a folder: the first Analyze starts sorting at once."""
    try:
        get_clf("accurate")
    except Exception as e:  # noqa: BLE001
        print(f"model preload skipped: {e}")


def bind_drop(win, url):
    """Drag a folder (or any file inside it) onto the window to analyze it. Needs the native window for full paths."""
    import json as _json
    try:
        if not (win.get_current_url() or "").startswith(url):
            return
        from webview.dom import DOMEventHandler

        def on_drop(e):
            paths = [f.get("pywebviewFullPath") for f in e.get("dataTransfer", {}).get("files", [])]
            paths = [Path(p) for p in paths if p]
            folder = next((str(p if p.is_dir() else p.parent) for p in paths), None)
            win.evaluate_js(f"droppedFolder({_json.dumps(folder)})")

        doc = win.dom.document
        doc.events.dragover += DOMEventHandler(lambda e: None, True, False)
        doc.events.drop += DOMEventHandler(on_drop, True, False)
        print("drag and drop ready")
    except Exception as e:  # noqa: BLE001
        print(f"drag and drop unavailable: {e}")
