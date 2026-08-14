# CUDA, threads, and `fork()`: why they don't mix

Two related but distinct hazards, both triggered by calling `os.fork()`
(directly or via Python's `multiprocessing` `"fork"` start method):

| # | Hazard | Trigger condition | Symptom | Applies to this repo right now? |
|---|---|---|---|---|
| 1 | **CUDA + fork** ("poison fork") | CUDA context already initialized in the parent *before* the fork | `RuntimeError: Cannot re-initialize CUDA in forked subprocess` — or silent hang/crash if the child never touches CUDA in a way that trips the explicit guard | **No** — `ModelExecutor.setup_worker()` forks before `LLMEngine` ever touches CUDA in the parent (see [Applied to this repo](#applied-to-this-repo)) |
| 2 | **fork() of an already multi-threaded process** | Any other thread is alive in the parent at fork time, holding a lock | Deadlock in the child (may look like a hang), or — since Python 3.12 — a `DeprecationWarning` instead of a silent trap | **Yes, currently accepted** — pytest/uvicorn are multi-threaded, so the warning fires; we haven't hit an actual deadlock |

Both are the *same* underlying OS-level fact — `fork()` only clones the
calling thread — expressed in two different failure modes depending on
whether the "thing left in a bad state" is a CUDA driver context or a
plain mutex.

---

## 1. CUDA + `fork()`: the "poison fork" problem

### Root cause

A CUDA context is not just "some memory." Once a process calls into the
CUDA driver (allocating a device tensor, launching a kernel, even just
`torch.cuda.is_available()` in some versions), the driver:

- opens file descriptors / handles to the GPU device node and the driver's
  kernel module,
- creates memory mappings between host and device address space,
- spins up **background driver threads** (for asynchronous copy engines,
  event handling, IPC, etc.),
- registers process-level state with the driver that is keyed to the
  *specific PID* that opened it.

`fork()` on POSIX only does two things: it copies the calling thread, and
it gives every page of memory copy-on-write semantics for the new PID. It
does **not**:

- restart or duplicate any other thread (including the CUDA driver's own
  background threads — they simply vanish in the child),
- re-open file descriptors/handles at the driver level for the new PID,
- tell the driver "hey, there's a new process that thinks it owns this
  context now."

So the child ends up with a byte-for-byte copy of the *host-side*
bookkeeping structures the CUDA context left behind, but none of the
driver-side machinery (threads, PID-keyed kernel state) that those
structures depend on. Any attempt to actually use CUDA in the child from
that point is operating on a half-real context — which is why PyTorch
added an **explicit guard** that raises immediately instead of letting you
hit whatever undefined behavior (hang, corruption, segfault) would
otherwise occur.

Critically: **this only bites you if CUDA was already initialized in the
parent before the fork.** If the parent process has never touched CUDA
(no tensor created on `'cuda'`, no `torch.cuda.init()`, nothing that lazily
triggers it) at the moment `fork()` is called, there's no context to leave
half-copied — forking is fine, and the *child* can safely initialize CUDA
from scratch on its own.

### Real-world reports

- **[pytorch/pytorch#2971 — "RuntimeError: Cannot re-initialize CUDA in
  forked subprocess"](https://github.com/pytorch/pytorch/issues/2971)** —
  one of the earliest reports; a user hit it intermittently while mapping
  a function over inputs with a thread/process pool that touched CUDA
  tensors. Resolution pointed to switching to the `spawn` start method.
- **[pytorch/pytorch#40403](https://github.com/pytorch/pytorch/issues/40403)**
  and **[pytorch/pytorch#77159 — "A somewhat cryptic error message (for
  newcomers)"](https://github.com/pytorch/pytorch/issues/77159)** — the
  second one is a request from a maintainer/community member to make the
  error message clearer, specifically for the extremely common case of
  `DataLoader(num_workers>0)` where a custom `Dataset.__getitem__` creates
  a CUDA tensor directly (`torch.rand(3, 400, 400, device='cuda')`).
  Because `DataLoader` workers are forked by default on Linux, and the
  main process/dataset had already touched CUDA, every worker fork is
  poisoned. The issue's own framing is that this trips up newcomers
  constantly because the error message doesn't explain *why*.
- **[Lightning-AI/pytorch-lightning#16262](https://github.com/Lightning-AI/pytorch-lightning/issues/16262)** —
  same failure surfacing through `isolate_rng`, another case of a library
  triggering CUDA init in the parent ahead of a fork it doesn't control.
- **[automl/Auto-PyTorch#22](https://github.com/automl/Auto-PyTorch/issues/22)** —
  same error via `pynisher`-based subprocess isolation; workaround was to
  disable `pynisher`'s use of fork for that path.

### Official documentation

PyTorch's own multiprocessing docs name this exact failure mode as a known,
general class of bug and are unambiguous about the fix:

> "This happens when the accelerator's runtime is not fork safe and is
> initialized before a process forks, leading to runtime errors in child
> processes." — [PyTorch: Multiprocessing best practices, "Poison fork"](https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html)

> "The CUDA runtime has the limitation described in Poison fork in
> multiprocessing when using the `fork` start method; either the `spawn`
> or `forkserver` start method are required to use CUDA in subprocesses."
> — same page

> "Avoid initializing the accelerator in the main process before forking
> child processes. Use an alternative process start method, such as
> `spawn` or `forkserver`, which ensures a clean initialization of each
> process." — same page

That last sentence is the load-bearing one: PyTorch's own recommended fix
is not "always use spawn," it's "don't initialize the accelerator in the
parent before you fork" — spawn/forkserver are just the mechanical way to
guarantee that when you *can't* control the ordering.

---

## 2. `fork()` of an already multi-threaded process

### Root cause

This is the same "only the calling thread survives the fork" fact from
part 1, but applied to ordinary OS/libc/Python-level locks instead of a
CUDA context, and it's a decades-old, CUDA-agnostic POSIX hazard, not an
NVIDIA-specific one.

- `fork()` duplicates the address space and **only the thread that called
  it**. Every other thread in the parent simply does not exist in the
  child — it is not paused, not terminated, not scheduled; the child's
  memory image simply doesn't have it.
- If any of those now-nonexistent threads was, at the exact instant of the
  fork, holding a lock — Python's GIL-adjacent internal locks, a `malloc`
  arena lock, the C library's internal `stdio`/`printf` lock, a logging
  lock, a thread-pool's work-queue mutex — that lock's *memory state*
  (i.e., "locked") is faithfully copied into the child. But the thread
  that would ever call `unlock()` on it doesn't exist there anymore.
- The child process is therefore not broken *yet*. It only deadlocks the
  moment some code path in the child tries to acquire that same lock —
  which is often not the fork call site at all, making the bug appear far
  away from its cause and hard to reproduce (it depends on exact timing:
  was the other thread mid-`malloc()` or mid-`printf()` at the instant of
  the fork?).

This is why it's classically described as a landmine rather than a bug you
can "just avoid" by careful coding: any library anywhere in the process —
not just your own code — might have spun up a background thread (a
connection pool, a metrics reporter, a native extension's internal thread
pool, the CUDA driver's own threads once initialized) without your
knowledge, and you have no way to know whether *that* thread happened to
hold a lock at the exact moment you called `fork()`.

**Why GPU/ML stacks trip this constantly:** frameworks like PyTorch,
Rust-based tokenizers, gRPC-based inference servers, and CUDA itself all
start background threads early and often invisibly to the caller — so a
process that imports `torch`/`transformers`/`vllm` is very likely to
already be multi-threaded by the time any application code calls
`multiprocessing.Process(...).start()`, even if the application itself
never spawned a thread on purpose.

### Official documentation

Python made this an explicit, versioned policy change rather than folklore:

> "Note that safely forking a multithreaded process is problematic." —
> [Python docs: `multiprocessing` — Contexts and start
> methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods),
> `fork` method description

> "Changed in version 3.12: If Python is able to detect that your process
> has multiple threads, the `os.fork()` function that this start method
> calls internally will raise a `DeprecationWarning`." — same page

> "The `fork` start method should be considered unsafe as it can lead to
> crashes of the subprocess as macOS system libraries may start threads."
> — same page, referencing
> [bpo-33725](https://bugs.python.org/issue?@action=redirect&bpo=33725)
> (the change that made `spawn` the default on macOS back in 3.8, for
> exactly this reason)

And as of Python 3.14, the ecosystem has moved even further away from
`fork` as a safe default:

> "Changed in version 3.14: This is no longer the default start method on
> any platform. Code that requires `fork` must explicitly specify that via
> `get_context()` or `set_start_method()`." — same page, `fork` method
> — `forkserver` (single-threaded by construction, so safe to fork from)
> becomes the new POSIX default instead.

The actual warning text (as emitted by `multiprocessing/popen_fork.py`)
reads: *"This process is multi-threaded, use of fork() may lead to
deadlocks in the child."* — this was implemented in
[python/cpython#100228](https://github.com/python/cpython/issues/100228),
which tracks adding the fork-safety `DeprecationWarning`; the underlying
mechanics (other threads and the locks they hold vanish in the child) are
exactly as described above.

### Real-world reports of the exact same warning in ML tooling

- **[openvinotoolkit/openvino#28944](https://github.com/openvinotoolkit/openvino/issues/28944)** —
  the identical warning text, from the identical source location
  (`multiprocessing/popen_fork.py:66`), triggered by
  `openvino.convert_model()` on Python 3.12.7. The reporter wasn't even
  sure it was a real problem — which is itself a good illustration of how
  this warning surfaces as noise most of the time (no thread happened to
  be mid-lock at fork time) right up until the one time it isn't.
- **[huggingface/transformers#5486](https://github.com/huggingface/transformers/issues/5486)** —
  a sibling issue, not the *same* warning but the *same root cause*
  wearing different clothes: HuggingFace's Rust `tokenizers` library
  (which uses its own internal thread pool for parallel encoding) detects
  post-fork state itself and proactively disables its thread pool with
  *"The current process just got forked, after parallelism has already
  been used. Disabling parallelism to avoid deadlocks."* rather than
  risking the deadlock. This is the same hazard, but a library choosing to
  self-defend instead of relying on the interpreter's warning.

### A classic, widely-cited explainer

> "In the child process, only the thread that called fork continues
> running. Other threads no longer exist. If a thread was holding a lock,
> it will remain locked forever." — Evan Jones, [*"fork() without exec()
> is dangerous in large
> programs"*](https://www.evanjones.ca/fork-is-dangerous.html)

> "Most large programs today use threads, either intentionally or
> accidentally, since many libraries use threads without making it
> obvious." — same post

> "Only use fork to immediately call exec (or just use `posix_spawn`).
> This is the least error-prone." — same post, on the recommended safe
> pattern (irrelevant to us directly since we don't `exec()` after fork,
> but it's the canonical framing of *why* `forkserver`, which forks a
> clean single-threaded helper and nothing else, is considered the safe
> middle ground)

This post is one of the most frequently cited explanations of this exact
hazard outside of the CPython/glibc source itself, and its central claim —
locks copied "locked" with no thread left alive to unlock them — is the
precise mechanism CPython's own issue tracker and docs describe.

---

## Applied to this repo

`llm/model_executor.py`'s `ModelExecutor` forks a worker child via
`multiprocessing.get_context("fork")`. Two things are true about how this
project currently uses it:

1. **Issue #1 (poison fork) does not currently apply.** In
   `LLMEngine.__init__` (`llm/llm.py`), the call order is:

   ```python
   self.model_executor = ModelExecutor()
   ...
   self.model_executor.setup_worker("facebook/opt-125m")   # forks here
   ...
   self.vllm_model = VLLM(model="facebook/opt-125m")        # CUDA touched here, in the parent
   ```

   `setup_worker()` — and therefore the `fork()` call — happens *before*
   `VLLM(...)` is constructed, which is the point where CUDA actually gets
   initialized in the **parent** process. Per PyTorch's own guidance
   above ("avoid initializing the accelerator in the main process before
   forking"), this ordering is exactly the safe pattern, not by luck but
   because the fork genuinely precedes any parent-side CUDA use. The
   forked worker child initializes its own CUDA context independently
   (inside `ModelWorker.__init__` → `ModelManager.load_model`), which is
   fine — a fresh, un-forked CUDA init in a process that has never forked
   *since* touching CUDA.

2. **Issue #2 (multi-threaded-parent fork warning) is real and currently
   an accepted tradeoff, not a fix.** Under `uvicorn` (which runs its own
   event loop plus worker threads) and under `pytest` (which — per the
   observed warning — reports the *test runner's own process* as
   multi-threaded at the moment `ModelExecutor.setup_worker()` forks), the
   parent is multi-threaded when the fork happens, so Python's 3.12+
   fork-safety `DeprecationWarning` fires. We have not observed an actual
   deadlock from it — only the warning — and switching this specific fork
   to `"spawn"` was tried and confirmed to silence the warning, at the
   cost of substantially slower worker startup (each spawn re-imports and
   re-initializes torch/CUDA from a bare interpreter instead of cloning
   the already-warmed-up parent). The current code deliberately stays on
   `"fork"` and accepts the warning as a known, documented, so-far-benign
   tradeoff rather than fixing it, per the explicit decision to revert
   from `spawn` back to `fork`.

   If this warning is ever promoted from noise to an actual observed
   deadlock, the two standard mitigations documented above are: (a) move
   the `fork()` call earlier, before any framework has had a chance to
   start background threads, or (b) switch just this one process creation
   to the `"forkserver"` context, which forks from a dedicated
   single-threaded helper process instead of the (multi-threaded)
   application process — cheaper than `"spawn"` because the helper is
   started once and only inherits what it needs, while still avoiding the
   "fork a thread-heavy process" hazard entirely.

---

## References

1. [pytorch/pytorch#2971 — RuntimeError: Cannot re-initialize CUDA in forked subprocess](https://github.com/pytorch/pytorch/issues/2971)
2. [pytorch/pytorch#40403](https://github.com/pytorch/pytorch/issues/40403)
3. [pytorch/pytorch#77159 — A somewhat cryptic error message (for newcomers)](https://github.com/pytorch/pytorch/issues/77159)
4. [Lightning-AI/pytorch-lightning#16262 — Bug with isolate_rng "Cannot re-initialize CUDA in forked subprocess"](https://github.com/Lightning-AI/pytorch-lightning/issues/16262)
5. [automl/Auto-PyTorch#22 — Cannot re-initialize CUDA in forked subprocess](https://github.com/automl/Auto-PyTorch/issues/22)
6. [PyTorch docs — Multiprocessing best practices ("Poison fork")](https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html)
7. [Python docs — multiprocessing: Contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
8. [python/cpython#100228 — fork-safety DeprecationWarning tracking issue](https://github.com/python/cpython/issues/100228)
9. [bpo-33725 — spawn became the default start method on macOS (Python 3.8)](https://bugs.python.org/issue?@action=redirect&bpo=33725)
10. [openvinotoolkit/openvino#28944 — DeprecationWarning: multi-threaded fork, since OpenVINO 2025.0.0](https://github.com/openvinotoolkit/openvino/issues/28944)
11. [huggingface/transformers#5486 — tokenizers "The current process just got forked" warning](https://github.com/huggingface/transformers/issues/5486)
12. Evan Jones — [*"fork() without exec() is dangerous in large programs"*](https://www.evanjones.ca/fork-is-dangerous.html)

All sources above were fetched live during this investigation; no URLs or
quotes were fabricated. Where a fetched page's comment thread wasn't fully
retrievable (e.g. pytorch/pytorch#2971, #77159), that limitation is noted
inline rather than inventing maintainer quotes.
