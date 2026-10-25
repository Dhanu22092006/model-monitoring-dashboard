# Jenkins Freestyle CI/CD guide (Windows) — `model-monitoring-dashboard-freestyle`

> **Status: written but NOT executed.** This guide was authored without access to Jenkins, Windows or Docker Desktop.
> Every command is plain `cmd.exe` batch with explicit exit-code checks, but you must run the job once and fix any
> environment-specific issue (paths, proxy, antivirus). No Jenkinsfile, Groovy, Declarative or Scripted Pipeline is used anywhere.

## 0. Prerequisites on the Jenkins machine (Windows 10/11 or Server)
| Need | Notes |
|---|---|
| Jenkins LTS (Windows service or `java -jar`) | Java 17+. |
| Git for Windows | `git --version` must work **for the Jenkins service account** (check *Manage Jenkins → System Information → PATH*). |
| Python **3.11 or 3.12** (64-bit, "Add to PATH") | pandas 3 / numpy 2.4 need ≥ 3.11. |
| Docker Desktop (WSL2 backend) running | The Jenkins service account must be able to reach the Docker engine: either run Jenkins as your logged-in user, or add the service account to the `docker-users` local group, then restart Jenkins. Docker Desktop must be started (enable *Start Docker Desktop when you sign in*). |
| `curl.exe` | Ships with Windows 10 1803+. |

**Plugins** (Manage Jenkins → Plugins → Available): *Git*, *GitHub* (webhook trigger), *Credentials Binding*, *JUnit*, *Workspace Cleanup* (optional), *Timestamper* (optional). *Pipeline* plugins are **not** needed.

## 1. Credentials (Manage Jenkins → Credentials → System → Global → Add)
| ID | Kind | Purpose |
|---|---|---|
| `mmd-secret-key` | Secret text | Value for `DJANGO_SECRET_KEY` (generate: `python -c "import secrets;print(secrets.token_urlsafe(50))"`). |
| `mmd-admin-password` | Secret text | First-run dashboard admin password (`ADMIN_PASSWORD`). |
| `github-pat` | Username with password (username = GitHub user, password = fine-grained PAT with *Contents: read*) | Only if the repo is **private**. Public repos need no credential. |

Never put these values in the repo, `.env.example`, or the job's plain-text fields.

## 2. Create the job
1. *New Item* → name **`model-monitoring-dashboard-freestyle`** → **Freestyle project** → OK.
2. **General**: description; tick **Discard old builds** → *Max # of builds to keep* = `20`, *Days* = `30`. (Optional) tick *GitHub project* and paste the repo URL.
3. **Source Code Management → Git**
   - Repository URL: `https://github.com/<you>/model-monitoring-dashboard.git`
   - Credentials: `github-pat` (private repos) or *- none -*
   - Branches to build → **Branch Specifier: `*/main`**
4. **Build Triggers** (pick one):
   - **GitHub hook trigger for GITScm polling** + in GitHub: *Repo → Settings → Webhooks → Add webhook* → Payload URL `https://<public-jenkins-host>/github-webhook/`, content type `application/json`, event *Just the push event*. Jenkins must be reachable from the internet (use a tunnel such as a reverse proxy; `localhost` is not reachable by GitHub).
   - **or Poll SCM** with schedule `H/5 * * * *` (works behind a firewall; no webhook needed).
5. **Build Environment**: tick **Use secret text(s) or file(s)** and add two bindings:
   `MMD_SECRET_KEY` ← `mmd-secret-key`, `MMD_ADMIN_PASSWORD` ← `mmd-admin-password`. Optionally tick *Add timestamps to the Console Output*.
6. **Build Steps** — add **10 separate** *Execute Windows batch command* steps with the scripts below, in order. Freestyle stops at the first step that returns non-zero, so **deployment (step 9) can never run after a failed validation.**
   Because only the *last* command's exit code is returned by a batch file, every command is followed by an explicit `if errorlevel 1 exit /b 1`.
7. **Post-build Actions**:
   - **Publish JUnit test result report** → Test report XMLs: `reports/*.xml` (tick *Allow empty results* off).
   - **Archive the artifacts** → `reports/**, logs/**` (tick *Archive even if build fails*; leave *fingerprint* off).
8. Save → **Build Now**.

> Each batch step is its own `cmd` process, so variables are re-set in every step that needs them.

### Step 1 — Verify Git, Python, Docker
```bat
@echo on
git --version
if errorlevel 1 exit /b 1
python --version
if errorlevel 1 exit /b 1
docker --version
if errorlevel 1 exit /b 1
docker compose version
if errorlevel 1 exit /b 1
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker engine is not reachable. Start Docker Desktop and check the Jenkins service account.
  exit /b 1
)
curl.exe --version >nul 2>&1
if errorlevel 1 exit /b 1
if not exist reports mkdir reports
if not exist logs mkdir logs
exit /b 0
```

### Step 2 — Isolated test environment + dependencies
```bat
@echo on
if exist .venv-ci rmdir /s /q .venv-ci
python -m venv .venv-ci
if errorlevel 1 exit /b 1
.venv-ci\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv-ci\Scripts\python.exe -m pip install -r requirements-ci.txt
if errorlevel 1 exit /b 1
exit /b 0
```

### Step 3 — Django system check
```bat
@echo on
set DJANGO_SECRET_KEY=%MMD_SECRET_KEY%
set DJANGO_DEBUG=False
.venv-ci\Scripts\python.exe manage.py check
if errorlevel 1 exit /b 1
exit /b 0
```

### Step 4 — Migration consistency (creates nothing)
```bat
@echo on
set DJANGO_SECRET_KEY=%MMD_SECRET_KEY%
.venv-ci\Scripts\python.exe manage.py makemigrations --check --dry-run
if errorlevel 1 (
  echo Model changes exist without a committed migration. Run makemigrations locally and commit.
  exit /b 1
)
exit /b 0
```

### Step 5 — Django unit + integration tests
```bat
@echo on
set DJANGO_SECRET_KEY=%MMD_SECRET_KEY%
set TEST_OUTPUT_DIR=%WORKSPACE%\reports
.venv-ci\Scripts\python.exe manage.py test automl.tests.test_django_app --noinput --testrunner xmlrunner.extra.djangotestrunner.XMLTestRunner > logs\django-tests.log 2>&1
set RC=%ERRORLEVEL%
type logs\django-tests.log
exit /b %RC%
```

### Step 6 — ML training / evaluation / prediction / drift tests
```bat
@echo on
set DJANGO_SECRET_KEY=%MMD_SECRET_KEY%
set TEST_OUTPUT_DIR=%WORKSPACE%\reports
.venv-ci\Scripts\python.exe manage.py test automl.tests.test_ml_services --noinput --testrunner xmlrunner.extra.djangotestrunner.XMLTestRunner > logs\ml-tests.log 2>&1
set RC=%ERRORLEVEL%
type logs\ml-tests.log
exit /b %RC%
```

### Step 7 — Build the Docker image (tagged by build number and commit)
```bat
@echo on
set SHORT=%GIT_COMMIT:~0,7%
docker build -t model-monitoring-dashboard:%BUILD_NUMBER% -t model-monitoring-dashboard:%SHORT% . > logs\docker-build.log 2>&1
set RC=%ERRORLEVEL%
type logs\docker-build.log
exit /b %RC%
```

### Step 8 — Docker smoke test (throw-away container on port 8012, no persistent volumes touched)
```bat
@echo on
set SMOKE=mmd-smoke-%BUILD_NUMBER%
docker rm -f -v %SMOKE% >nul 2>&1
docker run -d --name %SMOKE% -p 8012:8000 -e DJANGO_SECRET_KEY=%MMD_SECRET_KEY% -e ADMIN_USERNAME=smoke -e ADMIN_PASSWORD=%MMD_ADMIN_PASSWORD% model-monitoring-dashboard:%BUILD_NUMBER%
if errorlevel 1 exit /b 1
curl.exe -fsS --retry 20 --retry-delay 2 --retry-connrefused --retry-all-errors http://localhost:8012/health/
if errorlevel 1 goto fail
curl.exe -fsS -o NUL http://localhost:8012/accounts/login/
if errorlevel 1 goto fail
docker rm -f -v %SMOKE%
exit /b 0
:fail
docker logs %SMOKE% > logs\smoke-container.log 2>&1
type logs\smoke-container.log
docker rm -f -v %SMOKE%
exit /b 1
```
(`-v` removes only this container's *anonymous* volumes; the named volumes `mmd_data`, `mmd_artifacts`, `mmd_datasets` are never touched.)

### Step 9 — Deploy the validated image (backup → replace → verify → auto-rollback)
```bat
@echo on
set DJANGO_SECRET_KEY=%MMD_SECRET_KEY%
set ADMIN_USERNAME=admin
set ADMIN_PASSWORD=%MMD_ADMIN_PASSWORD%

rem --- remember the image that is serving now (empty on first deploy)
set PREV_IMAGE=
for /f "delims=" %%i in ('docker inspect --format "{{.Config.Image}}" model-monitoring-dashboard 2^>nul') do set PREV_IMAGE=%%i
echo Previous image: [%PREV_IMAGE%]

rem --- consistent SQLite backup inside the persistent data volume (only if a container exists)
if not "%PREV_IMAGE%"=="" (
  docker exec model-monitoring-dashboard python -c "import sqlite3,os;os.makedirs('/app/data/backups',exist_ok=True);s=sqlite3.connect('/app/data/db.sqlite3');d=sqlite3.connect('/app/data/backups/pre-build-%BUILD_NUMBER%.sqlite3');s.backup(d);d.close();s.close()"
  if errorlevel 1 (
    echo Database backup failed - aborting deploy, current app left running.
    exit /b 1
  )
)

set MMD_IMAGE_TAG=%BUILD_NUMBER%
docker compose up -d --no-build
if errorlevel 1 goto rollback
curl.exe -fsS --retry 30 --retry-delay 2 --retry-connrefused --retry-all-errors http://localhost:8002/health/
if errorlevel 1 goto rollback
docker tag model-monitoring-dashboard:%BUILD_NUMBER% model-monitoring-dashboard:known-good
if errorlevel 1 exit /b 1
echo Deployed build %BUILD_NUMBER%.
exit /b 0

:rollback
echo DEPLOY FAILED - collecting logs and rolling back
docker logs --tail 200 model-monitoring-dashboard > logs\deploy-failure.log 2>&1
type logs\deploy-failure.log
if "%PREV_IMAGE%"=="" (
  echo No previous image to roll back to.
  exit /b 1
)
for /f "tokens=2 delims=:" %%t in ("%PREV_IMAGE%") do set PREV_TAG=%%t
set MMD_IMAGE_TAG=%PREV_TAG%
docker compose up -d --no-build
curl.exe -fsS --retry 30 --retry-delay 2 --retry-connrefused --retry-all-errors http://localhost:8002/health/
echo Rolled back to model-monitoring-dashboard:%PREV_TAG%
exit /b 1
```
The new container is built **and smoke-tested (step 8) before** the running one is touched. Persistent volumes are never removed (`docker compose down -v` is never used). The job still ends *red* after a rollback so the failure is visible.

### Step 10 — Verify the deployment through the health endpoint
```bat
@echo on
curl.exe -fsS http://localhost:8002/health/ > logs\health.json
if errorlevel 1 exit /b 1
type logs\health.json
docker inspect --format "image={{.Config.Image}} status={{.State.Status}} health={{.State.Health.Status}}" model-monitoring-dashboard
if errorlevel 1 exit /b 1
exit /b 0
```

## 3. Manual rollback to the last known-good image
```powershell
$env:DJANGO_SECRET_KEY = "<same key>"; $env:MMD_IMAGE_TAG = "known-good"
docker compose up -d --no-build
curl.exe -fsS http://localhost:8002/health/
```
To restore a database backup: `docker compose stop web`, copy a file from the `mmd_data` volume's `backups/` over `db.sqlite3` (e.g. with a throw-away `docker run --rm -v model-monitoring-dashboard_mmd_data:/d alpine cp ...`), then `docker compose start web`.

## 4. Troubleshooting (Windows)
| Symptom | Fix |
|---|---|
| `'docker' is not recognized` in Jenkins only | Service account PATH differs. Add `C:\Program Files\Docker\Docker\resources\bin` to the *system* PATH and restart the Jenkins service. |
| `error during connect ... dockerDesktopLinuxEngine` / `Cannot connect to the Docker daemon` | Docker Desktop is not running or the account lacks access: start Docker Desktop, add the account to `docker-users`, re-login/restart Jenkins. |
| `python` opens the Microsoft Store | Turn off *App execution aliases* for python.exe, or use the full path of Python 3.12 in steps 2. |
| Step "passes" despite an error | A batch file returns only the last command's code — keep every `if errorlevel 1 exit /b 1`. |
| `%GIT_COMMIT%` empty | The step ran without SCM checkout; ensure Git is configured in SCM. |
| Port 8002 / 8012 in use | `netstat -ano \| findstr :8002`, stop the process or `docker ps` for stray containers. |
| Webhook shows 403/404 | URL must end with `/github-webhook/`; Jenkins must be public; if CSRF protection proxy compatibility is an issue, use Poll SCM instead. |
| `curl: (22)` on health | The container is unhealthy: `docker logs model-monitoring-dashboard`. Common causes: missing `DJANGO_SECRET_KEY`, `ALLOWED_HOSTS` lacking `127.0.0.1`. |
| Long path / antivirus locks in workspace | Enable long paths, exclude the workspace from real-time scanning, keep `.venv-ci` recreated each run (step 2). |
| Pip cannot download | Configure the proxy for the Jenkins service (`HTTPS_PROXY`) or use an internal mirror. |
