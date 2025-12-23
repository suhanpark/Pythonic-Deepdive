# Pythonic Deep Dive — Advanced Backend, AI, and Production Practices

## Table of Contents

1. [**Advanced Backend Foundations**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-1-backend-in-python)  
2. [**JIT (Just-In-Time Compilation in Python)**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-2-jit-just-in-time-compilation-in-python)
3. [**Decorators**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-3-decorators-in-python-basic--advanced)
4. [**Collections Library**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-4-collections-library-deep-dive)
5. [**Classes (Advanced Python OOP)**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-5-advanced-classes-in-python)
6. [**Typing (Advanced Python Type Hints)**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-5-advanced-classes-in-python)
7. [**Combined Production Example**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-7-combined-production-example-jit--decorators--typing--classes--collections)
8. [**Multithreading & Parallelism**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-7-combined-production-example-jit--decorators--typing--classes--collections)
9. [**Magic (Dunder) Methods and Lambdas**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-9-magicdunder-methods-and-lambdas)
10. [**PEP8 Guidelines**](https://github.com/suhanpark/Pythonic-Deepdive/blob/main/readme.md#section-10-pep8-guidelines)
11. [**Triton Inference Server**](url)
12. [**TorchServe**](url)
13. [**GraphQL**](url)
14. [**Kafka & Pub/Sub**](url)
15. [**WebSockets**](url)
16. [**gRPC**](url)

---
# **Section 1**: Backend in Python

This section covers a deep dive into building production-grade backends in Python. We go beyond simple CRUD apps and focus on concurrency, async programming, background tasks, deployment, and scaling strategies.


## 1. Async IO (concurrency without threads)

### Key Concepts
- Event loop schedules coroutines cooperatively.
- `await` yields control instead of blocking.
- Best for I/O-bound workloads (network, file, DB).

### Example: async HTTP calls
```python
import asyncio, httpx

async def fetch(url):
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        return len(r.text)

async def main():
    urls = ["https://example.com"] * 5
    results = await asyncio.gather(*(fetch(u) for u in urls))
    print(results)

asyncio.run(main())
````


## 2. Context Managers

### Why?

* Ensure resources are released (files, locks, DB connections).
* `with` handles enter/exit automatically.

### Example: custom context manager

```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        import time
        print(f"elapsed: {time.time() - self.start:.2f}s")

with Timer():
    do_work()
```


## 3. Queues

### Thread-safe producer/consumer

```python
import queue, threading, time

q = queue.Queue()

def producer():
    for i in range(5):
        q.put(i)
        time.sleep(0.1)

def consumer():
    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            break
        print("got", item)
        q.task_done()

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
```

### Async Queue

```python
import asyncio

async def worker(q):
    while True:
        item = await q.get()
        if item is None:
            break
        print("processing", item)

async def main():
    q = asyncio.Queue()
    await q.put(1)
    await worker(q)

asyncio.run(main())
```


## 4. Background Jobs & Scheduling

### Lightweight background tasks in FastAPI

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def send_email(email: str):
    print(f"Email sent to {email}")

@app.post("/signup")
async def signup(email: str, background: BackgroundTasks):
    background.add_task(send_email, email)
    return {"msg": "User created"}
```

### Celery with Redis/RabbitMQ (heavy duty)

```python
from celery import Celery

celery = Celery("tasks", broker="redis://localhost:6379/0")

@celery.task
def add(x, y):
    return x + y
```

Run worker:

```bash
celery -A tasks worker --loglevel=info
```

Call async:

```python
add.delay(3, 4)
```


## 5. Serving Models at Scale

### Options

* **FastAPI + ONNX Runtime / TorchScript** for simple cases.
* **Triton Inference Server** for multi-model GPU workloads.
* **TorchServe** for PyTorch production deployment.
* Batch requests, use async IO, and process pools for CPU/GPU-heavy work.


## 6. Deployment at Scale

### Containerization

* Use Docker to package app with dependencies.
* Multi-stage builds to keep images small.

### CI/CD

* GitHub Actions, GitLab CI, or Jenkins.
* Run tests, lint (flake8, black, mypy), and auto-deploy on merge.

### Kubernetes

* Horizontal Pod Autoscaler for scaling.
* Liveness/readiness probes.
* ConfigMaps and Secrets for env vars.

### Observability

* Logging: structlog or standard logging with JSON format.
* Metrics: Prometheus client (`Counter`, `Histogram`).
* Tracing: OpenTelemetry.


# **Section 2: JIT (Just-In-Time compilation in Python)**
---
## JIT in Python (Numba, PyPy, GPU kernels)

When folks say “JIT in Python,” they usually mean **Numba** (JIT to native via LLVM) or **PyPy** (a JITting Python runtime). There’s also JIT inside frameworks (e.g., `torch.compile`). For general Python, Numba is the workhorse.


## 0. When JIT Makes Sense
- Tight numeric loops, array math, simulations, feature engineering, kernels you can’t express nicely in NumPy.
- Speedups of 10–1000× without leaving Python.
- Hot path uses numbers/arrays, not arbitrary Python objects.


## 1. Numba Basics (CPU)

### Key knobs
- `@numba.njit`: compile to machine code, no Python objects allowed.
- `parallel=True`: auto-parallelize loops; use `numba.prange`.
- `fastmath=True`: allow algebraic reordering (tiny FP inaccuracy, big speed).
- `cache=True`: persist compiled artifact on disk.

```python
import numba as nb
import numpy as np

@nb.njit(cache=True, fastmath=True)
def l2_norms(a: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        s = 0.0
        row = a[i]
        for j in range(row.shape[0]):
            s += row[j] * row[j]
        out[i] = s ** 0.5
    return out
````

### Parallel loops

```python
@nb.njit(parallel=True, fastmath=True)
def row_means(X):
    n, m = X.shape
    out = np.empty(n, dtype=X.dtype)
    for i in nb.prange(n):  # parallel
        s = 0.0
        for j in range(m):
            s += X[i, j]
        out[i] = s / m
    return out
```


## 2. Advanced Numba: `vectorize`, `guvectorize`

### Elementwise ufunc

```python
@nb.vectorize(['float32(float32, float32)'], target='parallel')
def fused(a, b):
    return a * 0.5 + b * 0.5
```

### Generalized ufunc

```python
@nb.guvectorize(
    ['void(float32[:], float32[:], float32[:])'],
    '(n),(n)->()', target='parallel'
)
def dot_rows(x, w, out):
    acc = 0.0
    for i in range(x.shape[0]):
        acc += x[i] * w[i]
    out[0] = acc
```


## 3. GPU with Numba (CUDA)

```python
from numba import cuda
import numpy as np

@cuda.jit
def saxpy(a, x, y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = a * x[i] + y[i]

n = 1_000_000
x = np.ones(n, np.float32); y = np.ones(n, np.float32)
out = np.empty_like(x)
threads = 256
blocks = (n + threads - 1) // threads
saxpy[blocks, threads](2.0, x, y, out)
```

GPU tips:

* Use `cuda.shared.array` for shared memory tiles.
* Prefer contiguous arrays for coalesced loads.
* Batch enough work per kernel to amortize overhead.


## 4. PyPy JIT

* PyPy is a JITting interpreter for Python.
* Speeds up object-heavy pure Python code.
* Great for algorithmic workloads, but NumPy support is weaker.
* Try: `pypy3 your_script.py`.


## 5. JIT + NumPy vs “just NumPy”

* If you can vectorize in NumPy, do that first.
* If vectorization is awkward (awkward broadcasting, huge temporaries), JITed loop can be faster and clearer.
* Often best: hybrid — use NumPy for bulk ops, JIT the tricky inner loop.


## 6. Debugging & Pitfalls

* Use `numba.typeof(obj)` and read compile errors carefully.
* Avoid Python objects in `@njit` functions.
* RNG: limited support inside Numba; generate outside.
* Exceptions: prefer validation outside the kernel.
* First call cost: JIT happens on first call; warm up once.


## 7. Real Examples

### A. Fast cosine similarity

```python
@nb.njit(parallel=True, fastmath=True)
def cosine_sim(A, B):
    N, D = A.shape; M = B.shape[0]
    out = np.empty((N, M), dtype=np.float32)
    An = np.empty(N, np.float32); Bn = np.empty(M, np.float32)
    for i in nb.prange(N):
        s = 0.0
        for d in range(D): s += A[i, d]*A[i, d]
        An[i] = s**0.5
    for j in nb.prange(M):
        s = 0.0
        for d in range(D): s += B[j, d]*B[j, d]
        Bn[j] = s**0.5
    for i in nb.prange(N):
        for j in range(M):
            s = 0.0
            for d in range(D):
                s += A[i, d] * B[j, d]
            out[i, j] = s / (An[i]*Bn[j] + 1e-12)
    return out
```

### B. Rolling window mean

```python
@nb.njit
def rolling_mean(x, w):
    n = x.size
    out = np.empty(n - w + 1, dtype=x.dtype)
    s = 0.0
    for i in range(w): s += x[i]
    out[0] = s / w
    for i in range(w, n):
        s += x[i] - x[i-w]
        out[i - w + 1] = s / w
    return out
```

### C. Parallel histogram

```python
@nb.njit(parallel=True)
def hist_fixed(x, lo, hi, bins):
    counts = np.zeros(bins, np.int64)
    scale = bins / (hi - lo)
    tl_counts = np.zeros((nb.get_num_threads(), bins), np.int64)
    for i in nb.prange(x.size):
        t = int((x[i] - lo) * scale)
        if 0 <= t < bins:
            tl_counts[nb.threading.get_thread_id(), t] += 1
    for t in range(tl_counts.shape[0]):
        for b in range(bins):
            counts[b] += tl_counts[t, b]
    return counts
```


## 8. Numba vs Cython vs Native Extensions

* **Numba**: quick, no build step, great for numeric kernels.
* **Cython**: good for fine control, C/C++ integration.
* **Rust/C++ extensions**: max performance/control, heavy investment.

Rule of thumb: prototype with Numba → port to Cython/Rust if needed.


## 9. JIT in Async/Services

* Compile once at process start.
* Call JITed functions inside `ProcessPoolExecutor` for CPU-bound work.
* Avoid blocking event loop with JIT calls.


## 10. Benchmark Pattern

```python
import time, numpy as np

def bench(fn, *args, warmups=3, iters=10):
    for _ in range(warmups): fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters): fn(*args)
    dt = time.perf_counter() - t0
    return (dt/iters)*1000

A = np.random.randn(10000, 128).astype(np.float32)
B = np.random.randn(2000, 128).astype(np.float32)
print("cosine_sim ms:", bench(cosine_sim, A, B))
```


## 11. Checklist

* [ ] Profile first.
* [ ] Convert inputs to NumPy arrays with right dtype.
* [ ] Use `@njit`, `parallel=True`, `prange`.
* [ ] Validate vs NumPy baseline.
* [ ] Warmup at startup.
* [ ] For GPU, batch enough work.
* [ ] Keep Python objects out of JIT kernels.
---
# **Section 3**: Decorators in Python (Basic → Advanced)

Decorators are functions that take another function (or class) and return a modified function (or class). They are used for cross-cutting concerns like logging, caching, access control, retries, metrics, etc.


## 1. Basic Decorators

### Simple function decorator
```python
def log_calls(fn):
    def wrapper(*args, **kwargs):
        print(f"Calling {fn.__name__} with {args} {kwargs}")
        result = fn(*args, **kwargs)
        print(f"Returned {result}")
        return result
    return wrapper

@log_calls
def add(x, y): return x + y
````


## 2. Preserving Metadata

Use `functools.wraps` to keep function name, docstring, annotations.

```python
import functools

def log_calls(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"Calling {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper
```


## 3. Decorators with Arguments

```python
def retry(times: int = 3):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1} failed: {e}")
            raise
        return wrapper
    return decorator

@retry(times=5)
def flaky(): ...
```


## 4. Class-based Decorators

```python
class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.fn(*args)
        return self.cache[args]

@Memoize
def fib(n): ...
```


## 5. Decorating Classes

```python
def add_repr(cls):
    def __repr__(self): return f"{cls.__name__}({self.__dict__})"
    cls.__repr__ = __repr__
    return cls

@add_repr
class Point:
    def __init__(self, x, y): self.x, self.y = x, y
```


## 6. Stacking Decorators

```python
@timeit
@retry(times=3)
def work(x): ...
```

* Closest to the function executes first.


## 7. Async Decorators

```python
def log_async(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        print("before")
        result = await fn(*args, **kwargs)
        print("after")
        return result
    return wrapper

@log_async
async def fetch(): ...
```


## 8. Parameterized for Sync/Async

```python
from typing import TypeVar, Callable, Awaitable, Any
import inspect, functools

F = TypeVar("F", bound=Callable[..., Any])

def log_any(fn: F) -> F:
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def aw(*a, **k): print("async start"); return await fn(*a, **k)
        return aw  # type: ignore
    else:
        @functools.wraps(fn)
        def sw(*a, **k): print("sync start"); return fn(*a, **k)
        return sw  # type: ignore
```


## 9. Advanced: Decorating Methods

* Remember methods receive `self` or `cls`.
* Descriptor protocol means decorators on methods still work.

```python
def log_method(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        print(f"{self}.{fn.__name__}")
        return fn(self, *a, **k)
    return wrapper

class Service:
    @log_method
    def run(self): ...
```


## 10. Real Production Uses

* **Caching**: `functools.lru_cache`
* **Validation**: e.g., check user roles before running
* **Metrics**: increment Prometheus counters
* **Retries**: exponential backoff
* **Transaction boundaries**: start/commit/rollback DB


## 11. Pitfalls

* Don’t swallow exceptions silently.
* Don’t break signatures (use `functools.wraps`).
* Don’t overuse: readability first.
* Be careful stacking: order matters.


## 12. Checklist

* [ ] Always use `@functools.wraps`.
* [ ] Parameterize via nested functions.
* [ ] Separate sync/async if needed.
* [ ] Limit decorator logic: keep core function testable.
* [ ] Prefer small, composable decorators.
---
# **Section 4**: Collections Library (Deep Dive)

The `collections` module in Python provides high-performance container datatypes beyond the built-in `list`, `dict`, `set`, and `tuple`. Knowing when and how to use them is critical for clean, pythonic, and performant code.

## 1. Counter

- A dict subclass for counting hashable objects.

```python
from collections import Counter

words = ["apple", "banana", "apple", "orange", "banana", "apple"]
c = Counter(words)
print(c)  # Counter({'apple': 3, 'banana': 2, 'orange': 1})
---
# Most common
print(c.most_common(2))  # [('apple', 3), ('banana', 2)]
````

**Use Cases**:

* Frequency analysis.
* Top-k problems.
* Word counts, histograms.

## 2. defaultdict

* Like `dict` but provides a default value for missing keys.

```python
from collections import defaultdict

dd = defaultdict(list)
dd["a"].append(1)
dd["a"].append(2)
print(dd)  # {'a': [1, 2]}
```

**Use Cases**:

* Grouping items.
* Avoid `if key not in dict` boilerplate.

## 3. deque

* Double-ended queue.
* O(1) append/pop from both ends.

```python
from collections import deque

dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
print(dq)  # deque([0, 1, 2, 3, 4])
dq.pop()       # right
dq.popleft()   # left
```

**Use Cases**:

* Implement stacks/queues.
* Sliding window problems.
* BFS traversal.

## 4. namedtuple

* Tuple with named fields.

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(1, 2)
print(p.x, p.y)  # 1 2
```

**Use Cases**:

* Lightweight objects without class boilerplate.
* Immutable records.

## 5. OrderedDict

* Dict subclass that remembers insertion order.
* Since Python 3.7+, regular dicts also preserve insertion order.
* Still useful for methods like `move_to_end`.

```python
from collections import OrderedDict

od = OrderedDict()
od["a"] = 1
od["b"] = 2
od.move_to_end("a")
print(od)  # OrderedDict([('b', 2), ('a', 1)])
```

## 6. ChainMap

* Groups multiple dicts together.

```python
from collections import ChainMap

defaults = {"theme": "light", "show": True}
user = {"theme": "dark"}
cm = ChainMap(user, defaults)
print(cm["theme"])  # dark
print(cm["show"])   # True
```

**Use Cases**:

* Layered configs (user overrides defaults).
* Nested scopes.

## 7. UserDict, UserList, UserString

* Wrappers around dict/list/str for easy subclassing.

```python
from collections import UserDict

class MyDict(UserDict):
    def __setitem__(self, key, value):
        print(f"Setting {key}={value}")
        super().__setitem__(key, value)

d = MyDict()
d["x"] = 10
```

## 8. Putting it Together: Sliding Window with deque

```python
from collections import deque

def moving_average(iterable, n=3):
    it = iter(iterable)
    d = deque([], maxlen=n)
    s = 0
    for x in it:
        d.append(x)
        s += x
        if len(d) == n:
            yield s / n
            s -= d[0]

print(list(moving_average([40, 30, 50, 46, 39, 44], 3)))
```

## 9. Checklist

* [ ] Use `Counter` for counting/frequency.
* [ ] Use `defaultdict` for grouping.
* [ ] Use `deque` for fast append/pop both ends.
* [ ] Use `namedtuple` for immutable lightweight records.
* [ ] Use `ChainMap` for layered configs.
* [ ] Subclass `UserDict/List/String` instead of built-ins when customizing.

---
# **Section 5**: Advanced Classes in Python

Python classes go beyond simple containers — you can use dataclasses, slots, descriptors, mixins, ABCs, and even metaclasses to build robust, production-grade systems.

## 1. Dataclasses

- Automatically generate `__init__`, `__repr__`, `__eq__`, etc.
- Great for plain data objects.

```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    is_admin: bool = False

u = User(1, "Alice")
print(u)
````

### Options

* `frozen=True`: makes it immutable (hashable).
* `slots=True`: reduces memory footprint.
* `order=True`: generates ordering methods.

## 2. Slots

* Prevents dynamic attribute creation.
* Saves memory by avoiding `__dict__`.

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y): self.x, self.y = x, y

p = Point(1,2)---
# p.z = 3  # AttributeError
```

* Use when you create many small objects.
* Combine with `@dataclass(slots=True)` for convenience.

## 3. Properties

```python
class Celsius:
    def __init__(self, temp=0): self._temp = temp
    @property
    def temp(self): return self._temp
    @temp.setter
    def temp(self, v):
        if v < -273: raise ValueError("below absolute zero")
        self._temp = v
```

* Provides controlled access with attribute syntax.

## 4. Descriptors

* Reusable attribute access logic.

```python
class Positive:
    def __set_name__(self, owner, name): self.name = name
    def __get__(self, obj, objtype=None): return obj.__dict__[self.name]
    def __set__(self, obj, value):
        if value < 0: raise ValueError("must be >=0")
        obj.__dict__[self.name] = value

class Product:
    price = Positive()
    def __init__(self, price): self.price = price
```

## 5. Mixins

* Classes meant to be mixed into others to add behavior.
* Should be small, focused, no `__init__`.

```python
class JsonMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class User(JsonMixin):
    def __init__(self, id, name): self.id, self.name = id, name

print(User(1,"Bob").to_json())
```

## 6. Abstract Base Classes (ABCs)

* Enforce interface contracts.

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def add(self, item): ...
    @abstractmethod
    def get(self, id): ...

class MemoryRepo(Repository):
    def __init__(self): self.data = {}
    def add(self, item): self.data[item.id] = item
    def get(self, id): return self.data[id]
```

## 7. Metaclasses

* Classes of classes.
* Control class creation.
* Rare but powerful (ORMs, frameworks).

```python
class Singleton(type):
    _instances = {}
    def __call__(cls, *a, **k):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*a, **k)
        return cls._instances[cls]

class DB(metaclass=Singleton): pass

db1, db2 = DB(), DB()
assert db1 is db2
```

## 8. Multiple Inheritance and MRO

* Python uses C3 linearization to resolve method order.

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B,C): pass

print(D.mro())
```

* Always design mixins to avoid `__init__` conflicts.

## 9. Special Methods Recap

* `__init__`: constructor.
* `__new__`: allocation (for immutables).
* `__call__`: make instance callable.
* `__enter__/__exit__`: context managers.
* `__iter__/__next__`: iteration.
* `__getitem__/__setitem__`: container access.
* `__eq__/__lt__/__hash__`: comparisons.
* `__repr__/__str__`: representations.

## 10. Production Patterns

* Dataclasses for configs & DTOs.
* Slots for high-volume objects.
* Properties for validation.
* Descriptors for reusable validation logic.
* Mixins for small orthogonal features.
* ABCs for interfaces.
* Metaclasses only if truly needed.

## 11. Checklist

* [ ] Default to dataclass for simple data.
* [ ] Use slots for memory-sensitive workloads.
* [ ] Use ABCs for plugin/driver architectures.
* [ ] Mixins for optional features.
* [ ] Avoid metaclasses unless designing a framework.

---
# **Section 6**: Typing in Python (Advanced, Production-Grade)

Typing in Python makes APIs predictable, improves IDE support, and helps catch bugs early. Python itself ignores most annotations at runtime, but tools like `mypy`, `pyright`, FastAPI, and Pydantic leverage them.

## 1. Static vs Runtime Typing

- **Static**: type checkers (mypy, pyright).
- **Runtime**: Python ignores types, but frameworks can enforce (e.g., Pydantic).
- Rule of thumb: types are for *readability + tooling*, not runtime enforcement.

## 2. Generics

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self): self._items: list[T] = []
    def push(self, item: T) -> None: self._items.append(item)
    def pop(self) -> T: return self._items.pop()

ints = Stack[int]()
ints.push(1)
````

## 3. Variance

* `TypeVar("T", covariant=True)`: read-only producers.
* `TypeVar("T", contravariant=True)`: write-only consumers.

## 4. ParamSpec and Callable

```python
from typing import Callable, ParamSpec, TypeVar
import functools

P = ParamSpec("P")
R = TypeVar("R")

def log_calls(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*a: P.args, **k: P.kwargs) -> R:
        print("call", fn.__name__, a, k)
        return fn(*a, **k)
    return wrapper
```

Preserves original function signatures in wrappers.

## 5. Protocol vs ABC

### Protocol (structural typing)

```python
from typing import Protocol

class SupportsLen(Protocol):
    def __len__(self) -> int: ...

def total_length(x: SupportsLen) -> int: return len(x)

total_length("hi")       # works
total_length([1,2,3])    # works
```

* No inheritance required.
* Great for duck typing with safety.

### ABC (nominal typing)

```python
from abc import ABC, abstractmethod

class Repo(ABC):
    @abstractmethod
    def add(self, item): ...
```

* Enforced via explicit subclassing.

## 6. Literal and Annotated

```python
from typing import Literal, Annotated

def move(direction: Literal["N","S","E","W"]): ...

def set_speed(v: Annotated[float, "m/s"]): ...
```

* `Literal`: restricts to specific values.
* `Annotated`: attach metadata (units, validators).

## 7. Self Type

```python
from typing import Self

class Builder:
    def set_x(self, x: int) -> Self: self.x = x; return self
    def set_y(self, y: int) -> Self: self.y = y; return self
```

* Enables fluent API chaining without hardcoding class names.

## 8. TypedDict and Dataclasses

```python
from typing import TypedDict

class UserDict(TypedDict):
    id: int
    name: str
    is_admin: bool
```

* Structured dicts for JSON/APIs.
* Use `dataclass`/`pydantic` for runtime validation.

## 9. Abstract Containers

* Prefer `Mapping[str, Any]` over `dict[str, Any]`.
* Prefer `Sequence[T]` over `list[T]`.
* Prefer `Iterable[T]` when iteration is enough.

Think in interfaces (`collections.abc`), not concrete containers.

## 10. Type Narrowing

```python
def f(x: int | str):
    if isinstance(x, int):
        return x + 1
    reveal_type(x)  # str here for type checker
```

## 11. Config Objects

```python
from dataclasses import dataclass

@dataclass
class Config:
    retries: int
    timeout: float
```

Better than `dict[str, Any]`.

## 12. Plugin Registry with Protocols

```python
from typing import Protocol

class Plugin(Protocol):
    def run(self, data: str) -> str: ...

plugins: dict[str, Plugin] = {}
```

Any object with `.run(str)->str` works.

## 13. Enums with Typing

```python
from enum import Enum

class Role(str, Enum):
    USER = "user"
    ADMIN = "admin"

def has_access(r: Role) -> bool: return r == Role.ADMIN
```

## 14. Tooling Workflow

* Enable strict mode:

  * `mypy --strict`
  * Pyright `"strict": true`
* Run in CI/pre-commit.
* Combine with pytest (`pytest --mypy`).

## 15. Typing Pitfalls

* `Any` spreads like a virus — avoid except at boundaries.
* `Optional[int]` is just `int | None` in 3.10+.
* Avoid over-specific types (prefer abstract interfaces).
* Don’t enforce business rules with types (do at runtime).

## 16. Checklist

* [ ] Use abstract containers (`Mapping`, `Sequence`).
* [ ] Use `Protocol` for duck-typed APIs.
* [ ] Use `ParamSpec` and `TypeVar` in decorators.
* [ ] Add `Self` for fluent APIs.
* [ ] Use `TypedDict`/`dataclass`/`pydantic` for structured data.
* [ ] Run `mypy --strict` in CI.
* [ ] Separate runtime validation from static typing.
---
#  **Section 7**: Combined Production Example (JIT + Decorators + Typing + Classes + Collections)

This section shows how all the advanced Python features (JIT, decorators, typing, classes, and collections) can be combined in a realistic, production-style mini-service.

## 1. Problem Statement

We want to build a **data processing service** that:

- Validates structured input (users and transactions).
- Uses collections for efficient counting and grouping.
- Leverages a JIT-compiled function for numeric heavy lifting.
- Provides decorators for logging and caching.
- Uses classes and typing for maintainable design.

## 2. Code Example

```python
import functools, time
from dataclasses import dataclass
from typing import Protocol, TypedDict, Self
from collections import Counter, defaultdict
import numpy as np
import numba as nb
---
# ------------------------------------# Typing---
# ------------------------------------class TransactionDict(TypedDict):
    user_id: int
    amount: float
---
# ------------------------------------# Data Model---
# ------------------------------------@dataclass(slots=True)
class User:
    id: int
    name: str
    is_vip: bool = False

    def set_vip(self, flag: bool = True) -> Self:
        self.is_vip = flag
        return self
---
# ------------------------------------# Decorators---
# ------------------------------------def log_calls(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[LOG] {fn.__name__} called with {args} {kwargs}")
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"[LOG] {fn.__name__} returned {result} in {time.perf_counter()-t0:.4f}s")
        return result
    return wrapper

def memoize(fn):
    cache = {}
    @functools.wraps(fn)
    def wrapper(*a):
        if a not in cache:
            cache[a] = fn(*a)
        return cache[a]
    return wrapper
---
# ------------------------------------# Collections Usage---
# ------------------------------------class TransactionBook:
    def __init__(self):
        self.transactions: list[TransactionDict] = []

    def add(self, t: TransactionDict):
        self.transactions.append(t)

    def totals_by_user(self):
        totals = defaultdict(float)
        for t in self.transactions:
            totals[t["user_id"]] += t["amount"]
        return dict(totals)

    def counts(self):
        return Counter(t["user_id"] for t in self.transactions)
---
# ------------------------------------# JIT Accelerated Function---
# ------------------------------------@nb.njit(cache=True, fastmath=True)
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    n = x.size
    out = np.empty(n - w + 1, dtype=x.dtype)
    s = 0.0
    for i in range(w):
        s += x[i]
    out[0] = s / w
    for i in range(w, n):
        s += x[i] - x[i-w]
        out[i - w + 1] = s / w
    return out
---
# ------------------------------------# Service Interface with Protocol---
# ------------------------------------class Processor(Protocol):
    def run(self, book: TransactionBook) -> float: ...

class AverageProcessor:
    @log_calls
    @memoize
    def run(self, book: TransactionBook) -> float:
        totals = list(book.totals_by_user().values())
        arr = np.array(totals, dtype=np.float64)
        if arr.size < 3:
            return float(arr.mean()) if arr.size else 0.0
        ma = moving_average(arr, 3)
        return float(ma[-1])
---
# ------------------------------------# Example Usage---
# ------------------------------------if __name__ == "__main__":
    users = [User(1,"Alice"), User(2,"Bob").set_vip()]
    book = TransactionBook()
    book.add({"user_id": 1, "amount": 120.0})
    book.add({"user_id": 2, "amount": 300.0})
    book.add({"user_id": 1, "amount": 80.0})
    book.add({"user_id": 2, "amount": 250.0})

    processor = AverageProcessor()
    result = processor.run(book)
    print("Final average:", result)
````

## 3. Highlights

* **Typing**: `TypedDict`, `Protocol`, `Self` improve clarity and safety.
* **Dataclass with slots**: memory-efficient user objects.
* **Decorators**: logging + caching without cluttering business logic.
* **Collections**: `defaultdict`, `Counter` simplify aggregation.
* **Numba JIT**: accelerates numeric calculations.
* **Protocol**: defines pluggable interface (`Processor`).

## 4. Checklist

* [ ] Start with clear data models (dataclass, TypedDict).
* [ ] Use collections for idiomatic grouping/counters.
* [ ] Decorators for cross-cutting concerns.
* [ ] JIT the heavy math paths.
* [ ] Separate business logic (Processor) behind Protocols.

---
#  **Section 8**: Multithreading and Parallelism in Python

Python has multiple ways to achieve concurrency and parallelism. Due to the Global Interpreter Lock (GIL), threads are best for I/O-bound tasks, while processes or native extensions are best for CPU-bound workloads.

## 1. The GIL Reality

- **Global Interpreter Lock (GIL)**: only one thread executes Python bytecode at a time.
- **I/O-bound** tasks → use `threading` or `asyncio`.
- **CPU-bound** tasks → use `multiprocessing`, `ProcessPoolExecutor`, or JIT/native extensions.

## 2. Threading

### Example
```python
import threading, time

def worker(n):
    print(f"start {n}")
    time.sleep(1)
    print(f"done {n}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
````

* Useful for parallel I/O waits.

## 3. ThreadPoolExecutor

High-level API for pooling threads.

```python
from concurrent.futures import ThreadPoolExecutor
import requests

def fetch(url: str) -> str:
    return requests.get(url).text

urls = ["https://example.com"] * 10
with ThreadPoolExecutor(max_workers=5) as pool:
    texts = list(pool.map(fetch, urls))
```

## 4. CPU-Bound with ProcessPoolExecutor

Bypasses GIL by using multiple processes.

```python
from concurrent.futures import ProcessPoolExecutor
import math

def cpu_heavy(n: int) -> float:
    return sum(math.sqrt(i) for i in range(n))

with ProcessPoolExecutor() as pool:
    results = list(pool.map(cpu_heavy, [10**6]*4))
```

## 5. Queues for Producer/Consumer

Thread-safe communication.

```python
import queue, threading, time
q = queue.Queue()

def producer():
    for i in range(5):
        q.put(i)
        time.sleep(0.1)

def consumer():
    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            break
        print("got", item)
        q.task_done()

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
```

## 6. multiprocessing Primitives

* `multiprocessing.Queue` and `Pipe` for IPC.
* `Value` and `Array` for shared memory.
* `multiprocessing.shared_memory` for efficient array sharing.

## 7. Best Practices

1. Prefer executors (`ThreadPoolExecutor`, `ProcessPoolExecutor`) over manual threads/processes.
2. Batch work — don’t spawn a thread per item.
3. Always guard `if __name__ == "__main__":` with multiprocessing.
4. Avoid shared mutable state — prefer queues or message passing.
5. Use `shutdown(wait=True)` on executors.
6. Combine with `asyncio.to_thread` for blocking calls inside async.

## 8. Example Use Cases

### A. Parallel Web Scraping (I/O-bound)

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def fetch(url: str) -> int:
    return len(requests.get(url).content)

urls = ["https://example.com"] * 20
with ThreadPoolExecutor(max_workers=10) as pool:
    lengths = list(pool.map(fetch, urls))
print(sum(lengths))
```

### B. Parallel Compute (CPU-bound)

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def simulate(n: int) -> float:
    x = np.random.randn(n)
    return float((x**2).mean())

with ProcessPoolExecutor() as pool:
    results = list(pool.map(simulate, [10_000_000]*4))
print(np.mean(results))
```

### C. Streaming Producer/Consumer

```python
import queue, threading, time
from collections import deque

q = queue.Queue()
window = deque(maxlen=5)

def producer():
    for i in range(20):
        q.put(i)
        time.sleep(0.05)

def consumer():
    while True:
        try: x = q.get(timeout=1)
        except queue.Empty: break
        window.append(x)
        print("avg", sum(window)/len(window))

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
```

## 9. Profiling & Scaling Tips

* Profile first; don’t parallelize prematurely.
* Threading overhead: microseconds–milliseconds. Worth it if tasks ≥ 10ms.
* Process overhead: ms–100ms. Worth it if tasks ≥ 0.5s.
* For large jobs: use **joblib** or **Ray**.

## 10. Checklist

* [ ] Use threads for I/O-bound workloads.
* [ ] Use processes for CPU-bound workloads.
* [ ] Use queues for safe communication.
* [ ] Executors > manual threads/processes.
* [ ] Benchmark vs single-thread baseline.

# **Section 9: Magic/Dunder Methods and Lambdas**

---
## Magic (Dunder) Methods and Lambda Functions

Magic methods (also called dunder methods because of the double underscores) allow your classes to hook into Python’s data model.  
Lambda functions provide anonymous, inline functions for small tasks.

## 1. Identity & Representation

```python
class User:
    def __init__(self, id: int, email: str): self.id, self.email = id, email
    def __repr__(self) -> str:  # unambiguous, for devs/logs
        return f"User(id={self.id!r}, email={self.email!r})"
    def __str__(self) -> str:   # user-facing
        return self.email
````

* Prefer `__repr__` over `__str__`.
* `__repr__` should be unambiguous, `__str__` user-friendly.

## 2. Truthiness and Size

```python
class Batch:
    def __init__(self, items): self._items = list(items)
    def __len__(self): return len(self._items)
    def __bool__(self): return bool(self._items)   # optional
```

* Implement `__len__` for truthiness by default.
* Add `__bool__` for special semantics.

## 3. Iteration

```python
class Range2:
    def __init__(self, start, stop): self.start, self.stop = start, stop
    def __iter__(self):
        cur = self.start
        while cur < self.stop:
            yield cur
            cur += 1
```

* Use generators inside `__iter__` for simplicity.

## 4. Containers and Slicing

```python
class Vector:
    def __init__(self, xs): self.xs = list(xs)
    def __len__(self): return len(self.xs)
    def __getitem__(self, key):
        if isinstance(key, slice): return Vector(self.xs[key])
        return self.xs[key]
    def __setitem__(self, i, v): self.xs[i] = v
    def __contains__(self, v): return v in self.xs
```

* Implement slice handling in `__getitem__`.
* Add `__reversed__` for efficient reverse iteration.

## 5. Arithmetic & Operator Overloading

```python
from math import hypot
class Vec2:
    __slots__ = ("x","y")
    def __init__(self,x,y): self.x,self.y=x,y

    def __add__(self, o): return Vec2(self.x+o.x, self.y+o.y)
    def __radd__(self, o): return self.__add__(o)      # for sum()
    def __iadd__(self, o): self.x+=o.x; self.y+=o.y; return self

    def __mul__(self, k: float): return Vec2(self.x*k, self.y*k)
    def __rmul__(self, k: float): return self.__mul__(k)

    def __abs__(self): return hypot(self.x, self.y)
```

* Implement reflected (`__r*__`) and in-place (`__i*__`) operators when natural.

## 6. Comparisons & Hashing

```python
from functools import total_ordering

@total_ordering
class Version:
    def __init__(self, major, minor): self.major, self.minor = major, minor
    def __eq__(self, o): return (self.major, self.minor) == (o.major, o.minor)
    def __lt__(self, o): return (self.major, self.minor) < (o.major, o.minor)
    def __hash__(self): return hash((self.major, self.minor))
```

* Equality and hash must align.
* Use `@total_ordering` to generate missing comparison methods.

## 7. Callable and Context Manager

```python
class TokenBucket:
    def __init__(self, rate, burst): self.rate, self.tokens = rate, burst
    def __call__(self, n=1) -> bool:
        if self.tokens >= n: self.tokens -= n; return True
        return False

class Resource:
    def __enter__(self): ...; return self
    def __exit__(self, exc_type, exc, tb): ...; return False
```

* `__call__` lets objects act like functions.
* `__enter__/__exit__` implement context managers.

## 8. Attribute Access Hooks

```python
class Lazy:
    def __init__(self): self._data = None
    def __getattr__(self, name):   # only called if attribute not found
        if name == "data":
            self._data = self._load(); return self._data
        raise AttributeError(name)
    def __getattribute__(self, name):  # intercepts *every* access
        return super().__getattribute__(name)
```

* Use `__getattr__` for lazy loading.
* Avoid `__getattribute__` unless absolutely needed.

## 9. Construction Lifecycle

* `__new__`: allocation (useful for immutables/singletons).
* `__init__`: initialization.
* `__init_subclass__`: runs on subclass creation.
* `__class_getitem__`: enables generic-like `MyClass[int]`.

## 10. Pickling & Copying

```python
def __getstate__(self): ...
def __setstate__(self, state): ...
def __copy__(self): ...
def __deepcopy__(self, memo): ...
```

## 11. Lambdas

### Basics

```python
f = lambda x: x * x
```

* Concise, inline anonymous functions.
* Expression-only (no statements, no annotations).

### Idiomatic Uses

* **Key functions**:

```python
sorted_users = sorted(users, key=lambda u: (u.last_login is None, u.last_login))
```

* **Mapping small transforms**:

```python
names = [s.strip().title() for s in raw_names]
```

* **Tiny predicates**:

```python
only_errors = [r for r in logs if (lambda c: 400 <= c < 600)(r.code)]
```

### Late-Binding Pitfall

```python
funcs = [lambda: i for i in range(3)]
[f() for f in funcs]    # [2, 2, 2]

funcs = [lambda i=i: i for i in range(3)]
[f() for f in funcs]    # [0, 1, 2]
```

* Lambdas close over variables by reference.
* Fix with default arg binding or `functools.partial`.

### Lambdas vs def

* Use `def` if function is non-trivial or reused.
* Use `lambda` for short, throwaway expressions.

## 12. Checklist

**Magic Methods**

* [ ] Implement only when natural for your type.
* [ ] `__repr__` accurate, `__hash__` only for immutables.
* [ ] Support slicing in `__getitem__` if sequence-like.
* [ ] Follow comparison/hash contracts.

**Lambdas**

* [ ] Keep them tiny; use `def` for bigger logic.
* [ ] Avoid late-binding bugs.
* [ ] Prefer comprehensions to `map`/`filter` with lambdas.

# **Section 10: PEP8 Guidelines**
---
## PEP8 Guidelines (Python Style Guide)

PEP8 is the official style guide for Python code. Following it makes your code readable, consistent, and professional.  

## 1. Philosophy

- Readability counts.  
- Consistency matters across a codebase.  
- Use tools (`black`, `ruff`, `flake8`, `isort`) to enforce automatically.  

## 2. Code Layout

### Indentation
- Use 4 spaces per indentation level (no tabs).
- Continuation lines should align with opening delimiter or use hanging indent.

```python---
# Good
def func(x, y, z):
    return (x + y
            + z)
---
# Good
def func(
    x, y, z
):
    return x + y + z
````

### Line Length

* Limit lines to 79 characters (72 for docstrings/comments).
* Tools like `black` default to 88 chars (common modern practice).

### Blank Lines

* 2 blank lines before top-level functions and classes.
* 1 blank line between class methods.
* 1 blank line to separate logical sections inside functions.

## 3. Imports

* One import per line.
* Order: standard library → third-party → local.
* Use absolute imports; avoid relative unless necessary.
* Avoid `from module import *`.

```python
import os
import sys

import numpy as np
import requests

from myapp import utils
```

## 4. Whitespace

* No extra spaces inside parentheses/brackets/braces.
* No space before a comma, semicolon, colon.
* One space after a comma.
* Spaces around operators, but not for keyword arguments.

```python---
# Good
x = [1, 2, 3]
y = func(a=1, b=2)
total = a + b - c
---
# Bad
x = [ 1 , 2 , 3 ]
y = func( a = 1 , b = 2 )
```

## 5. Naming Conventions

* **Variables & functions**: `lowercase_with_underscores`
* **Classes & exceptions**: `CapWords` (PascalCase)
* **Constants**: `ALL_CAPS_WITH_UNDERSCORES`
* **Private**: leading underscore `_internal`
* **Avoid**: names that clash with builtins (`list_`, `id_`)

## 6. Strings

* Either `'single'` or `"double"`, just be consistent.
* Triple quotes `"""docstring"""` for docstrings.

## 7. Docstrings & Comments

* Use docstrings for modules, classes, functions.
* First line: short summary. Optional longer description after blank line.
* Comments explain *why*, not just *what*.

```python
def connect(host: str, port: int) -> None:
    """Establish a TCP connection.

    Retries up to 3 times before raising ConnectionError.
    """
```

## 8. Functions & Classes

* Space after commas in parameters.
* No spaces around `=` in keyword arguments/defaults.
* One decorator per line.

```python
@timeit
@retry(times=3)
def compute(x: int, y: int = 0) -> int:
    return x + y
```

## 9. Expressions & Statements

* Don’t cram multiple statements with `;`.
* Avoid backslashes for line continuation; use parentheses.
* Use `is` / `is not` for `None` checks.
* Use `==` for equality.

```python
if foo is None:
    ...
if bar == 0:
    ...
```

## 10. Idiomatic Python

* Prefer list comprehensions over `map`/`filter` if clearer.
* Use `enumerate()` instead of `range(len(...))`.
* Use `with` for resource management.
* Don’t assign lambdas; use `def`.

```python---
# Good
for i, val in enumerate(items):
    ...

with open("file.txt") as f:
    data = f.read()
```

## 11. Tools

* **black** → auto-format.
* **isort** → sort imports.
* **flake8/ruff/pylint** → linting.
* Use pre-commit hooks to enforce before pushing.

## 12. Checklist

* [ ] 4 spaces per indent, 79-char lines.
* [ ] Group imports: stdlib, third-party, local.
* [ ] Spaces around operators, not inside brackets.
* [ ] snake\_case for funcs/vars, PascalCase for classes, UPPER for constants.
* [ ] `is None` / `is not None`.
* [ ] Triple-quoted docstrings.
* [ ] Use comprehensions and idioms.
* [ ] Run black + flake8 + isort.


---
# **Section 11**: Triton Inference Server (Deep Dive)

Triton Inference Server is NVIDIA’s production-grade inference serving system. It supports multiple frameworks, dynamic batching, observability, and deployment across GPUs/CPUs.

## 1. Why Triton?

- Multi-framework support:
  - TensorRT, PyTorch (TorchScript/ONNX), TensorFlow, ONNX Runtime, Python backend, OpenVINO.
- Dynamic micro-batching across requests → higher GPU utilization.
- Serve multiple models with versioning.
- Deployment on bare metal, containers, or Kubernetes.
- HTTP/gRPC protocols, Prometheus metrics.

## 2. Model Repository

Triton watches a **model repo** (local FS, S3, GCS).

```

models/
resnet50/
1/
model.onnx
config.pbtxt
unet/
1/
model.plan
config.pbtxt

````

### Minimal `config.pbtxt`
```protobuf
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input", data_type: TYPE_FP32, dims: [3,224,224] }
]
output [
  { name: "output", data_type: TYPE_FP32, dims: [1000] }
]
dynamic_batching { max_queue_delay_microseconds: 100 }
````

## 3. Running Triton

```bash
docker run --gpus=all --rm -it \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

* **8000** = HTTP
* **8001** = gRPC
* **8002** = metrics

## 4. Making Requests

### HTTP

```bash
curl -X POST http://localhost:8000/v2/models/resnet50/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{"name":"input", "shape":[1,3,224,224], "datatype":"FP32",
                "data":[...]}]
  }'
```

### gRPC (Python)

```python
import tritonclient.grpc as grpc
import numpy as np

client = grpc.InferenceServerClient("localhost:8001")
inp = grpc.InferInput("input", [1,3,224,224], "FP32")
data = np.random.rand(1,3,224,224).astype(np.float32)
inp.set_data_from_numpy(data)
out = grpc.InferRequestedOutput("output")

resp = client.infer("resnet50", inputs=[inp], outputs=[out])
print(resp.as_numpy("output"))
```

## 5. Advanced Features

### a) Ensemble Models

Chain multiple models/pipelines.

```protobuf
name: "pipeline"
platform: "ensemble"
input [ { name: "image", data_type: TYPE_UINT8, dims: [224,224,3] } ]
output [ { name: "class", data_type: TYPE_INT32, dims: [1] } ]

ensemble_scheduling {
  step [
    { model_name: "preprocess", input_map { key: "raw" value: "image" }
      output_map { key: "normed" value: "preprocessed" } },
    { model_name: "resnet50", input_map { key: "input" value: "preprocessed" }
      output_map { key: "output" value: "class" } }
  ]
}
```

### b) Concurrency Control

```protobuf
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0,1] }
]
```

* Deploy multiple model instances across GPUs.

### c) Multi-Model Ensemble

* Run vision + text models in one request.

## 6. Observability

* Prometheus at `:8002/metrics`:

  * `nv_inference_request_success`
  * `nv_inference_queue_duration_us`
  * `nv_inference_compute_input_duration_us`
  * `nv_inference_count`
* Combine with Grafana dashboards.

## 7. Deployment Patterns

* **Kubernetes**: Helm charts, custom YAML.
* **Model store**: S3/GCS bucket, hot reload.
* **Multi-tenancy**: one Triton per GPU pool.
* **LLM serving**: NVIDIA Triton-based micro-batching for LLMs.

## 8. When to Use Triton

✅ Multi-model workloads (CV + NLP + custom pipelines)
✅ Need dynamic batching for GPUs
✅ Framework-agnostic serving
✅ Strong observability (Prometheus)
✅ GPU-rich environment

❌ Simpler case with only one model → FastAPI + ONNX Runtime / TorchScript may suffice.

## 9. Python Backend

Write preprocessing/postprocessing in Python.

```
models/mymodel/1/model.py
models/mymodel/config.pbtxt
```

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for req in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(req, "input")
            arr = in_tensor.as_numpy()
            out_tensor = pb_utils.Tensor("output", arr * 2)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
```

## 10. Checklist

* [ ] Organize models with versioning.
* [ ] Write `config.pbtxt` for inputs/outputs/batching.
* [ ] Start server with `--model-repository`.
* [ ] Call via HTTP or gRPC.
* [ ] Monitor metrics with Prometheus.
* [ ] Enable dynamic batching for GPU efficiency.
* [ ] Use ensembles for preprocessing or multi-model flows.

# **Section 12: TorchServe**

---
# TorchServe (Deep Dive)

TorchServe is the official model serving framework for PyTorch. It helps you package, serve, and scale PyTorch models in production.

## 1. Why TorchServe?

- Purpose-built for PyTorch models.  
- REST and gRPC APIs.  
- Supports multi-model serving.  
- Easy to integrate with Kubernetes, Docker, or cloud platforms.  
- Custom handlers for preprocessing/postprocessing.  
- Model versioning, batch inference, and metrics.

## 2. Installing TorchServe

```bash
pip install torchserve torch-model-archiver
````

## 3. Packaging a Model

Use the **model archiver** tool (`.mar` file).

```bash
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --serialized-file resnet50.pt \
  --handler image_classifier \
  --extra-files index_to_name.json
```

* `--handler`: built-in (e.g., `image_classifier`, `text_classification`) or custom Python file.
* Produces `resnet50.mar`.

## 4. Starting TorchServe

```bash
torchserve --start --ncs --model-store model_store --models resnet50=resnet50.mar
```

* `--ncs`: no config snapshot.
* `--model-store`: directory of `.mar` files.
* `--models`: alias=model.

Stop with:

```bash
torchserve --stop
```

## 5. Making Requests

### Inference (REST)

```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet50 \
  -T test.jpg
```

### gRPC

TorchServe also provides a gRPC API on port 7070.

## 6. Custom Handlers

Create your own preprocessing, inference, postprocessing.

```python
from ts.torch_handler.base_handler import BaseHandler
import torch

class MyHandler(BaseHandler):
    def preprocess(self, data):
        return torch.tensor(data[0]["body"])
    def inference(self, x):
        return self.model(x).tolist()
    def postprocess(self, preds):
        return preds
```

Package with `--handler my_handler.py`.

## 7. Configuration

Config file `config.properties`:

```text
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
enable_envvars_config=true
model_store=model_store
default_workers_per_model=4
```

* Management API (`8081`) for loading/unloading models dynamically.
* Metrics API (`8082`) exposes Prometheus metrics.

## 8. Scaling & Deployment

* Run multiple workers per model (in config).
* Use Docker images for reproducible deployment.
* Integrate with **Kubernetes** via Helm charts.
* Autoscale with HPA + custom metrics.

## 9. Metrics & Logging

* Metrics exposed in Prometheus format.
* Logs:

  * `logs/model_log.log`
  * `logs/ts_log.log`

## 10. When to Use TorchServe

✅ You’re all-in on PyTorch
✅ Need custom preprocessing/postprocessing in Python
✅ Want an official, supported PyTorch serving stack
✅ Need management API to dynamically load/unload models

❌ Multi-framework serving (Triton is better)
❌ Ultra-low latency GPU batch serving (Triton excels more)

## 11. Checklist

* [ ] Package model with `torch-model-archiver`.
* [ ] Start TorchServe with `--model-store`.
* [ ] Call via REST or gRPC.
* [ ] Write custom handler for preprocessing/postprocessing.
* [ ] Tune workers per model.
* [ ] Expose Prometheus metrics.
* [ ] Use Docker/K8s for scaling.

# **Section 13: GraphQL**

---
# GraphQL (Deep Dive)

GraphQL is a query language and runtime for APIs. Unlike REST, clients specify exactly what data they need. It’s strongly typed and introspectable.

## 1. Why GraphQL?

- Avoids over-fetching and under-fetching.
- Schema-driven API → self-documenting.
- Single endpoint (`/graphql`) instead of many REST endpoints.
- Great for frontend teams that want flexibility.
- Strong tooling ecosystem (Apollo, Graphene, Strawberry, gql).

## 2. Core Concepts

- **Schema**: type system of the API.
- **Query**: read data.
- **Mutation**: write/update/delete data.
- **Subscription**: real-time updates over WebSockets.
- **Resolver**: function that fetches field data.

## 3. Example Schema

```graphql
type User {
  id: ID!
  name: String!
  email: String!
}

type Query {
  users: [User!]!
  user(id: ID!): User
}

type Mutation {
  createUser(name: String!, email: String!): User!
}
````

## 4. Example Queries

```graphql
query {
  users {
    id
    name
  }
}
```

```graphql
mutation {
  createUser(name: "Alice", email: "alice@example.com") {
    id
  }
}
```

## 5. Python Server Example (Strawberry + FastAPI)

```python
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class User:
    id: int
    name: str

@strawberry.type
class Query:
    @strawberry.field
    def users(self) -> list[User]:
        return [User(id=1, name="Alice"), User(id=2, name="Bob")]

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")
```

Run server → query via POST to `/graphql`.

## 6. GraphQL Subscriptions

* Typically use WebSockets.
* Example: real-time chat, notifications.

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def countdown(self, from_: int) -> int:
        for i in range(from_, 0, -1):
            yield i
```

## 7. GraphQL vs REST

| Feature             | REST              | GraphQL                |
| ------------------- | ----------------- | ---------------------- |
| Endpoints           | Many              | One (`/graphql`)       |
| Response shape      | Fixed             | Client-defined         |
| Over/Under fetching | Common            | Avoided                |
| Versioning          | Via new endpoints | Evolve schema          |
| Real-time           | Webhooks/polling  | Subscriptions (native) |

## 8. Best Practices

1. Design schema around business/domain, not DB tables.
2. Use batching (e.g., DataLoader) to avoid N+1 queries.
3. Use schema directives for auth, deprecation, validation.
4. Cache at resolver or query level.
5. Monitor query complexity (avoid expensive queries).

## 9. Deployment

* Serve over FastAPI, Flask, or ASGI.
* Apollo Gateway for federation across microservices.
* Combine with Pub/Sub for subscriptions.
* Protect with depth/complexity limits.

## 10. Checklist

* [ ] Define schema (`type`, `query`, `mutation`, `subscription`).
* [ ] Implement resolvers with batching.
* [ ] Validate and document with schema introspection.
* [ ] Support subscriptions for real-time features.
* [ ] Secure endpoints (auth, complexity analysis).
* [ ] Deploy with ASGI/FastAPI + Strawberry/Graphene.

# **Section 14: Kafka & Pub/Sub**
---
# Kafka & Pub/Sub (Deep Dive)

Both **Apache Kafka** and **Google Cloud Pub/Sub** are distributed messaging systems designed for event-driven and streaming architectures. They decouple producers and consumers, enabling scalable and reliable communication.

## 1. Why Kafka or Pub/Sub?

- Asynchronous, decoupled communication.  
- High throughput (millions of msgs/sec).  
- Replayability (Kafka keeps logs; Pub/Sub with retention).  
- Stream processing with Flink, Spark, Beam.  
- Backbone for event-driven microservices.

## 2. Kafka Core Concepts

- **Producer**: publishes messages.  
- **Consumer**: reads messages.  
- **Topic**: category of messages.  
- **Partition**: parallelism unit; each is an ordered log.  
- **Consumer Group**: multiple consumers balancing partitions.  
- **Broker**: Kafka server node.  

## 3. Pub/Sub Core Concepts

- **Publisher**: sends messages.  
- **Subscriber**: receives messages.  
- **Topic**: named resource to which messages are sent.  
- **Subscription**: named resource representing the stream of messages from a topic to subscribers.  
- **Ack**: subscriber must acknowledge processing to remove message.  

## 4. Kafka Example (Python with confluent-kafka)

### Producer
```python
from confluent_kafka import Producer

p = Producer({"bootstrap.servers": "localhost:9092"})
p.produce("test", key="k1", value="hello world")
p.flush()
````

### Consumer

```python
from confluent_kafka import Consumer

c = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "mygroup",
    "auto.offset.reset": "earliest"
})
c.subscribe(["test"])

while True:
    msg = c.poll(1.0)
    if msg is None: continue
    if msg.error(): print("Error:", msg.error()); continue
    print(f"Got {msg.key()}: {msg.value().decode()}")
```

## 5. GCP Pub/Sub Example (Python)

```python
from google.cloud import pubsub_v1
---
# Publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("my-project", "my-topic")
future = publisher.publish(topic_path, b"Hello world", key="k1")
print(future.result())
---
# Subscriber
subscriber = pubsub_v1.SubscriberClient()
sub_path = subscriber.subscription_path("my-project", "my-sub")
def callback(msg):
    print(f"Received {msg.data}")
    msg.ack()
subscriber.subscribe(sub_path, callback=callback)
```

## 6. Ordering & Delivery Semantics

* Kafka: messages in a partition are ordered; “at least once” by default; “exactly once” with idempotent producer + transactional consumer.
* Pub/Sub: ordering optional; “at least once” by default; “exactly once” with new Pub/Sub Lite or Dataflow integration.

## 7. Scaling Patterns

* Kafka partitions scale horizontally → throughput scales with partitions.
* Pub/Sub auto-scales transparently.
* Use consumer groups to parallelize processing.

## 8. Stream Processing

* Kafka Streams API, Flink, Spark → stateful event processing.
* GCP Dataflow integrates with Pub/Sub for ETL and ML pipelines.

## 9. Deployment

* Kafka:

  * Self-managed (ZooKeeper or KRaft mode).
  * Confluent Cloud (managed Kafka).
* Pub/Sub:

  * Fully managed by Google.
  * Global service, integrates with GCP IAM, BigQuery, Dataflow.

## 10. When to Use Which?

* **Kafka**

  * Complex event streaming pipelines.
  * Need fine-grained control over partitions, offsets, retention.
  * On-prem or hybrid-cloud.

* **Pub/Sub**

  * Simplicity, fully managed, GCP-native.
  * Don’t want to manage clusters.
  * Tight integration with BigQuery, Dataflow, GCS.

## 11. Checklist

* [ ] Define topics (partition count for Kafka).
* [ ] Choose consumer group strategy.
* [ ] Handle retries and dead-letter queues.
* [ ] Monitor lag and throughput.
* [ ] Secure with TLS, IAM, ACLs.
* [ ] Integrate with stream processors if needed.

---
# WebSockets (Deep Dive)

WebSockets provide a persistent, bidirectional communication channel between client and server, unlike HTTP which is request/response.

## 1. Why WebSockets?

- Real-time applications: chat, games, live dashboards, notifications.  
- Lower latency than polling/long-polling.  
- Persistent connection, full-duplex communication.  

## 2. Lifecycle

1. HTTP handshake (`Upgrade: websocket` header).  
2. Connection upgraded to WebSocket protocol.  
3. Both client and server can send messages anytime.  
4. Close frames shut down the connection.  

## 3. Python Example (FastAPI)

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        await ws.send_text(f"Echo: {data}")
````

### Client Example (JS)

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = (event) => console.log(event.data);
ws.onopen = () => ws.send("Hello");
```

## 4. Broadcasting

```python
from fastapi import WebSocket
from typing import List

clients: List[WebSocket] = []

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            for c in clients:
                await c.send_text(f"{data}")
    except:
        clients.remove(ws)
```

## 5. WebSocket vs Alternatives

| Feature    | Polling         | SSE (Server-Sent Events)         | WebSockets               |
| ---------- | --------------- | -------------------------------- | ------------------------ |
| Direction  | Client → Server | Server → Client (unidirectional) | Bidirectional            |
| Latency    | High            | Low                              | Low                      |
| Complexity | Low             | Medium                           | Medium                   |
| Use Cases  | Legacy          | Notifications, streaming         | Chat, games, collab apps |

## 6. Scaling WebSockets

* Requires sticky sessions (connection affinity) when using load balancers.
* For multiple servers, use **Pub/Sub backend** (Redis, Kafka, GCP Pub/Sub).
* Example: each server subscribes to Redis channel and forwards to its WS clients.

## 7. Security

* Always use `wss://` in production (TLS).
* Authenticate connection during handshake (JWT, API key).
* Implement rate limiting and message size caps.

## 8. Deployment Patterns

* Behind Nginx/Envoy with WebSocket upgrade support.
* Use ASGI servers (Uvicorn, Hypercorn, Daphne).
* Kubernetes: enable sticky sessions or use stateful routing.

## 9. Checklist

* [ ] Use `asyncio`/ASGI server for concurrency.
* [ ] Broadcast via shared backend (Redis, Kafka) if scaling.
* [ ] Authenticate on connect.
* [ ] Protect with TLS and rate limits.
* [ ] Test reconnection logic on clients.

# **Section 16: gRPC**

---
# gRPC (Deep Dive)

gRPC is a high-performance, open-source RPC framework by Google. It uses Protocol Buffers (Protobuf) for serialization and HTTP/2 for transport, enabling efficient, strongly typed APIs.

## 1. Why gRPC?

- Strongly typed contracts via `.proto` files.  
- Efficient binary serialization (Protobuf).  
- Built-in streaming (client, server, bidirectional).  
- Cross-language support (C++, Go, Java, Python, Rust, etc.).  
- Authentication, deadlines, retries built in.  
- Runs over HTTP/2 (multiplexed streams).  

## 2. Defining a Service

Example `service.proto`:

```proto
syntax = "proto3";

package example;

service UserService {
  rpc GetUser (UserRequest) returns (UserResponse);
  rpc ListUsers (Empty) returns (UserList);
  rpc StreamUsers (Empty) returns (stream UserResponse);
}

message UserRequest {
  int32 id = 1;
}

message UserResponse {
  int32 id = 1;
  string name = 2;
}

message UserList {
  repeated UserResponse users = 1;
}

message Empty {}
````

## 3. Generating Python Code

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service.proto
```

This generates:

* `service_pb2.py` (messages).
* `service_pb2_grpc.py` (service stubs).

## 4. Implementing the Server

```python
import grpc
from concurrent import futures
import service_pb2, service_pb2_grpc

class UserService(service_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        return service_pb2.UserResponse(id=request.id, name="Alice")
    def ListUsers(self, request, context):
        return service_pb2.UserList(users=[service_pb2.UserResponse(id=1, name="Bob")])
    def StreamUsers(self, request, context):
        for i in range(3):
            yield service_pb2.UserResponse(id=i, name=f"User {i}")

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
service_pb2_grpc.add_UserServiceServicer_to_server(UserService(), server)
server.add_insecure_port("[::]:50051")
server.start()
server.wait_for_termination()
```

## 5. Client Example

```python
import grpc
import service_pb2, service_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = service_pb2_grpc.UserServiceStub(channel)

resp = stub.GetUser(service_pb2.UserRequest(id=123))
print(resp)

for u in stub.StreamUsers(service_pb2.Empty()):
    print("Got", u)
```

## 6. Streaming Types

* **Unary RPC**: single request → single response.
* **Server streaming**: single request → stream of responses.
* **Client streaming**: stream of requests → single response.
* **Bidirectional streaming**: stream ↔ stream.

## 7. Deadlines & Metadata

```python
stub.GetUser(
    service_pb2.UserRequest(id=1),
    timeout=1.0,
    metadata=[("authorization", "Bearer token")]
)
```

* Deadlines enforce request timeouts.
* Metadata supports auth, tracing.

## 8. Authentication

* TLS: secure channels with `grpc.ssl_channel_credentials`.
* mTLS: client + server certificates.
* Integrates with OAuth/JWT.

## 9. Observability

* Interceptors for logging/tracing/metrics.
* Prometheus exporters.
* gRPC reflection: introspection tools (grpcurl).

## 10. Deployment

* Run behind Envoy or Istio for load balancing.
* Kubernetes: expose via `ClusterIP` or `Ingress`.
* Combine with service mesh for retries, circuit breaking.

## 11. When to Use gRPC

✅ High-performance microservice communication.
✅ Multi-language ecosystems.
✅ Need streaming RPCs.
✅ Strong type safety.

❌ External/public APIs (HTTP+JSON/GraphQL easier for consumers).
❌ Simple cases with no strict typing needs.

## 12. Checklist

* [ ] Write `.proto` with messages and services.
* [ ] Generate stubs via `protoc`.
* [ ] Implement server methods.
* [ ] Call via client stubs.
* [ ] Handle deadlines, metadata, retries.
* [ ] Secure with TLS/mTLS.
* [ ] Monitor with interceptors and Prometheus.

