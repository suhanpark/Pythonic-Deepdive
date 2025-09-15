# Deep Dive: Python Backend Engineering


## 1. Project Structure & Maintainability

### Why it matters

In personal projects, you can throw everything into main.py.
In real systems, you’ll need readable, modular, and testable code.

Example: Clean structure
```
app/
  __init__.py
  main.py        # FastAPI entrypoint
  api/
    __init__.py
    routes_user.py
    routes_auth.py
  core/
    config.py     # env settings
    security.py   # JWT, OAuth2
  models/
    user.py
    post.py
  services/
    user_service.py
  tests/
    test_user.py
```

### Config with Pydantic

```Python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

Now the app gets settings from env vars automatically, perfect for containers and CI/CD.

### Best Practices
* Use Poetry or pip-tools for dependency management.
* Enforce style: black, isort, flake8, mypy.
* Use pre-commit hooks so every commit is auto-formatted.

## 1. FastAPI: Async APIs at Scale

### Why it matters

FastAPI is industry standard for Python APIs: async, type-safe, self-documenting.

Example: Simple API

```Python
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    return {"id": user_id, "name": "Alice"}
```

Visit /docs → you get Swagger UI for free.

### Dependency Injection (DI)

```Python
from fastapi import Depends

def get_db():
    db = connect_to_db()
    try:
        yield db
    finally:
        db.close()

@app.get("/items")
async def read_items(db=Depends(get_db)):
    return db.query("SELECT * FROM items")
```

This makes testing & mocking trivial.

### Best Practices
* Organize routes in routers/ and mount with app.include_router.
* Handle auth with JWT/OAuth2 (fastapi.security).
* Add middleware for logging, error handling, request tracing.

## 1. Databases: SQLAlchemy 2.0 + Alembic
1. Databases: SQLAlchemy 2.0 + Alembic

### Why it matters

Most production backends are database-heavy. You’ll need ORM + migrations.

Example: SQLAlchemy ORM

```Python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
```

### Async with asyncpg

```Python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_user(user_id: int):
    async with SessionLocal() as session:
        return await session.get(User, user_id)
```

### Alembic Migration

```Bash
alembic init migrations
alembic revision --autogenerate -m "create users table"
alembic upgrade head
```

Now your DB schema is version-controlled.

## 4. Testing & Quality

### Why it matters

Industry code is tested, otherwise it doesn’t scale with teams.

Example: pytest

```Python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_get_user():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/users/1")
    assert resp.status_code == 200
    assert resp.json()["id"] == 1
```

### Fixtures

```Python
@pytest.fixture
def fake_user():
    return {"id": 1, "name": "Alice"}
```

Use fixtures for reusable test objects.

### Best Practices
* Aim for unit + integration tests, not 100% coverage.
* Use tox to test against multiple Python versions.
* Add tests into CI/CD pipeline.

## 5. Performance & Concurrency

### Why it matters

Users expect APIs to be fast. Python has asyncio, multiprocessing, and task queues.

Async Example

```Python
import asyncio
import httpx

async def fetch(url):
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        return r.text

async def main():
    results = await asyncio.gather(*[fetch("https://example.com") for _ in range(5)])
    print(results)

asyncio.run(main())
```

With async, 1000 requests/sec is possible on one server.

### Background tasks with Celery

```Python
from celery import Celery

app = Celery("tasks", broker="redis://localhost:6379/0")

@app.task
def add(x, y):
    return x + y
```

Use Celery for email, ML inference, video processing, etc.

## 6. Deployment & Ops

```Docker
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CI/CD (GitHub Actions)

```Yaml
name: Backend CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.12" }
      - run: pip install poetry
      - run: poetry install
      - run: pytest

Observability
* Logs: structlog or loguru.
* Metrics: Prometheus + Grafana.
* Tracing: OpenTelemetry.
```

## 7. Architecture Patterns
* Monoliths first → only split to microservices when needed.
* Repository pattern → clean DB access.
* Service layer → business logic separated from API.
* Event-driven → Kafka / PubSub when scaling to distributed systems.
* CQRS & Event Sourcing → for high-traffic, auditable systems (e.g., fintech).


## 8. Authentication & Authorization

### Why it matters

Every serious backend needs to secure data and APIs. “Login systems” in production involve tokens, OAuth2, sessions, and permissions.

### JWT (JSON Web Tokens) with FastAPI

```Python
from datetime import datetime, timedelta
from jose import JWTError, jwt

SECRET_KEY = "super-secret"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

OAuth2 with FastAPI

```Python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return {"user": payload["sub"]}
```

### Industry best practice:
* Access token (short-lived, JWT).
* Refresh token (long-lived, stored securely).
* Role-based or attribute-based access control.
* External providers (Google, GitHub, Okta).

## 9. Advanced Async & Concurrency Patterns

### Why it matters

Python’s async is powerful but tricky. Understanding when to use asyncio vs multiprocessing vs distributed tasks is key.

### Async: I/O-bound tasks

```Python
import asyncio, httpx

async def fetch(url):
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        return r.text
```

### Multiprocessing: CPU-bound tasks

```Python
from concurrent.futures import ProcessPoolExecutor

def heavy_compute(x):
    return sum(i*i for i in range(x))

with ProcessPoolExecutor() as executor:
    results = list(executor.map(heavy_compute, [10**6, 10**7]))
```

### Distributed tasks (Celery/RQ)

Use when tasks need persistence, retries, scaling across nodes.
* Email notifications, ML inference, video rendering.
* Celery with Redis/RabbitMQ = industry standard.

### Rule of thumb:
* I/O bound → asyncio.
* CPU bound → multiprocessing / numba / Cython.
* Cross-server tasks → Celery, Dramatiq, or Kafka.

## 10. Caching & Performance Optimization

### Why it matters

In production, DB calls and API calls are expensive. Caching = speed.

### In-memory cache

```Python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_config(key: str):
    return expensive_lookup(key)
```

### Redis cache

```Python
import redis

cache = redis.Redis()

def get_user(user_id):
    key = f"user:{user_id}"
    if data := cache.get(key):
        return data
    db_data = query_db(user_id)
    cache.set(key, db_data, ex=60)  # TTL 60s
    return db_data
```

### Best practices:
* Cache frequently-read queries.
* Use TTL (time-to-live).
* Invalidate on updates.
* For large-scale → Redis cluster / Memcached.

## 11. Observability: Logs, Metrics, Traces

### Why it matters

In production, when things go wrong, you need to know where, why, and how often.

### Structured logging

```Python
import structlog

log = structlog.get_logger()

log.info("user_login", user_id=123, ip="10.0.0.1")
```

### Metrics
* Expose Prometheus metrics via /metrics endpoint.
* Track request latency, DB query times, queue length.

### Tracing
* OpenTelemetry with Jaeger/Tempo → trace a request across microservices.
* Example: track a request through API → DB → task queue → external API.

* Logs tell you what happened.
* Metrics tell you how often it happens.
* Traces tell you where in the system it happens.


## 12. Security in Production
* Input validation (FastAPI + Pydantic already helps).
* Use HTTPS everywhere.
* Rate limiting (e.g., slowapi with Redis).
* SQL injection: use ORM params, never string concat.
* Secrets management: never commit .env; use Vault, GCP Secret Manager, AWS Secrets Manager.
* Security headers (CORS, CSP).
* Regular dependency checks (safety check, pip-audit).


## 13. Scaling Architectures

Monolith → Microservices → Event-driven
* Start with modular monolith (one codebase, well-structured).
* Break off services if needed:
* Auth service.
* Payments service.
* Notifications service.

Event-driven architecture
* Kafka / GCP PubSub → decouple services.
* Example:
* User signs up → “user.created” event → email service sends welcome mail.

Pro tip: Don’t jump to microservices until you must. Complexity is huge.


## 14. Deployment Patterns
* Containers: Docker best practices → small, reproducible builds.
* Orchestration: Kubernetes, Helm charts for config.
* Serverless: GCP Cloud Run / AWS Lambda for simple APIs.
* CI/CD: Automate build → test → deploy pipelines.

Canary releases
* Roll out new versions to 5% of traffic.
* Monitor errors.
* Gradually roll to 100%.


## 15. Design Patterns in Backend Python
* Repository pattern → isolate DB access.
* Service layer → business logic separate from API.
* Factory pattern → useful for creating objects with config/env.
* Observer/Event pattern → send signals on events (Django signals, or custom pub/sub).

Example: Repository pattern

```Python
class UserRepository:
    def __init__(self, session):
        self.session = session

    def get(self, user_id):
        return self.session.query(User).filter(User.id == user_id).first()

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def get_user_profile(self, user_id):
        return self.repo.get(user_id)
```


## 16. Domain-Driven Design (DDD) in Python

For large backend systems, DDD helps separate domain logic from infrastructure.

Structure:
```
app/
  domain/
    user/
      entities.py
      value_objects.py
      services.py
  infrastructure/
    repositories/
    db.py
  api/
    routes_user.py
```

This prevents your business rules from being tied to the database or framework.

## 17. Authentication & Authorization

### Why It Matters

Every production backend has to handle:
* Authentication: who you are.
* Authorization: what you can do.

Mistakes here → data leaks, broken apps, security holes.


### 1. The Common Industry Standards
* Session-based auth: server stores a session ID in DB/cache; cookie in browser references it. Old-school, stateful.
* JWT-based auth: client holds a signed token; server just validates signature. Stateless, scales horizontally.
* OAuth2 / OpenID Connect: standardized way to log in via Google, GitHub, etc.
* RBAC/ABAC: Role-Based or Attribute-Based Access Control.

Modern Python backends often start with JWT for APIs, and add OAuth2 if you need external logins.


### 2. JWT Auth in FastAPI

Install
```
pip install "python-jose[cryptography]" passlib[bcrypt]
```

Example: Generate & Verify Tokens

```Python
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext

SECRET_KEY = "super-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

FastAPI Integration

```Python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
def login(username: str, password: str):
    # Normally you'd verify against DB
    if username != "alice" or password != "password":
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return {"user": payload["sub"]}
```

Now your API has working JWT authentication.


### 3. Refresh Tokens & Expiry

Access tokens are short-lived (minutes). Refresh tokens are long-lived (days/weeks).
* Access token = in memory (client).
* Refresh token = in DB/Redis, revocable.

Flow:
1.	User logs in → gets access + refresh token.
2.	Access token expires.
3.	Client exchanges refresh token for new access token.
4.	Logout = delete refresh token from DB/Redis.


### 4. Role-Based Access Control (RBAC)
```Python
def has_role(required_role: str):
    def wrapper(user=Depends(read_users_me)):
        if user.get("role") != required_role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return wrapper

@app.get("/admin")
def read_admin(user=Depends(has_role("admin"))):
    return {"msg": "Welcome Admin!"}
```

Scale up with permission tables in DB, or attribute-based rules.


### 5. Best Practices in Production
* Never store plaintext passwords → always hash with bcrypt/argon2.
* Always use HTTPS in production.
* Store JWT secret in environment variables, not code.
* Use short access tokens + refresh tokens.
* Revoke refresh tokens on logout/password change.
* Rotate JWT signing keys periodically.
* Use fastapi.middleware.cors to secure APIs from unwanted origins.


### 6. When to Use OAuth2

If you want “Login with Google/GitHub/etc.” → use OAuth2/OpenID Connect.
* Libraries: authlib, python-social-auth.
* FastAPI integrates easily with Google, GitHub, Okta, Auth0.

# Security Deep Dive

## 1. Password Storage & User Data
* Always hash passwords with bcrypt or argon2.
* bcrypt is industry standard.
* argon2 is even stronger, with memory-hard resistance against GPU cracking.
* Never roll your own crypto.

Example:
```Python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```
* Store only the hash in DB, never plaintext.
* Add rate limiting on login attempts (to block brute force).


## 2. Transport Security
* Always enforce HTTPS → use Let’s Encrypt or cloud-managed SSL.
* Use HSTS headers:
```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

* Disable weak TLS versions/ciphers.


## 3. API Security Basics
* Use rate limiting to prevent abuse:
```Python
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import FastAPI

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.get("/login")
@limiter.limit("5/minute")
async def login():
    return {"msg": "Rate limited login"}
```

* Validate all inputs with Pydantic. Never trust the client.
* Strip HTML/JS from user input if it can be rendered later (avoid XSS).


## 4. JWT Security
* Keep access tokens short-lived (5–30 min).
* Keep refresh tokens long-lived but revocable (stored in Redis or DB).
* Use token blacklists if needed (to revoke compromised tokens).
* Rotate signing keys periodically (use JWKS if scaling).

* Example: Use kid (Key ID) in JWT header so clients know which signing key is valid.


## 5. Cross-Origin Resource Sharing (CORS)
* APIs often need to allow browsers from other domains (e.g., frontend on app.com, API on api.com).
* Use CORSMiddleware in FastAPI:
```Python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myfrontend.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```


* Never use allow_origins=["*"] in production unless it’s a public API.


## 6. CSRF Protection
* CSRF = attacker tricks user’s browser into making a request.
* Protection methods:
* Use SameSite cookies (SameSite=Lax or Strict).
* Use CSRF tokens for forms.
* For APIs with JWT in Authorization headers → generally safe.


## 7. Secrets Management
* Never hardcode secrets (JWT keys, DB passwords).
* Use environment variables locally.
* Use secret managers in production:
* AWS Secrets Manager
* GCP Secret Manager
* HashiCorp Vault

* Example: Load from GCP Secret Manager into FastAPI at startup.


## 8. Database Security
* Always use parameterized queries (ORM already does this).
* Restrict DB user permissions (don’t use root).
* Encrypt data at rest (Postgres TDE, MySQL InnoDB encryption).
* Encrypt data in transit (sslmode=require in Postgres).
* Rotate credentials periodically.


## 9. Logging & Sensitive Data
* Never log plaintext passwords, tokens, or PII.
* Mask sensitive fields before logging.

Example:
```Python
def sanitize_log(data: dict):
    hidden = {"password", "ssn", "credit_card"}
    return {k: ("***" if k in hidden else v) for k, v in data.items()}
```


## 10. Vulnerability Scanning & Dependency Safety
* Run pip-audit or safety on CI/CD:
```Bash
pip install pip-audit
pip-audit
```

* Keep dependencies pinned (poetry.lock, requirements.txt).
* Use Docker images with slim bases (e.g., python:3.12-slim).
* Regularly patch OS + libs.


## 11. Security Headers

Add headers to every response:
* Content-Security-Policy (prevent inline JS execution).
* X-Frame-Options: DENY (prevent clickjacking).
* X-Content-Type-Options: nosniff.
* Referrer-Policy: no-referrer.

* Can add with Starlette middleware.


## 12. Audit & Monitoring
* Log all auth events (login, logout, refresh).
* Alert on suspicious behavior (many failed logins, refresh token misuse).
* Track IP addresses and device fingerprints.


## 13. Pen Testing & Threat Modeling
* Use OWASP Top 10 as a checklist.
* Run penetration tests before production.
* Consider bug bounty programs if the app grows.
