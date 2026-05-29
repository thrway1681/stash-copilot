# Productionalization Plan

*Original plan provided by user for repo cleanup and organization.*

## Clean-up Principles

- Work in a feature branch
- Commit frequently
- Commit before / after every phase
- Update imports incrementally
- Run tests when necessary
- Embrace parallelism. Spin up as many parallel agents you deem necessary
- All diagrams should be saved as SVG
- All documents should be concise and informative. Avoid being overly verbose
- Add to project configs. Do not install any one-offs
- Avoid untyped variables and generic types
- Adhere to "Don't Repeat Yourself" principles
- Adhere to "Keep It Simple Stupid" principles
- Adhere to "You Aren't Gonna Need It" principles
- Attempt to minimize code complexity
- Create proper type definitions for complex data structures

## Phase 0: Pre-cleanup Checklist

- [x] Run any type checking. Note current state
- [x] Run test suite. Note current coverage % → See `docs/phase-0-checklist.md`
- [x] List all config / settings locations → See `docs/phase-0-checklist.md`
- [x] Note any performance-critical paths → See `docs/phase-0-checklist.md`

## Phase 1: Document Current State

1. [x] Gain a thorough understanding of the code → See `docs/phase-1/architecture.md`
2. [x] Document database schemas → See `docs/phase-1/database-schemas.md`
3. [x] Create a mermaid diagram of the CURRENT architecture → See `docs/phase-1/architecture.md`
4. [x] Create a table of features and their dependencies → See `docs/phase-1/features.md`
5. [x] Create a DESIGN.md that describes the aesthetic → See `docs/phase-1/DESIGN.md`
6. [x] Document features (all 16 task classes + frontend) → See `docs/phase-1/features.md`
7. [x] Document findings → All Phase 1 docs complete
8. [x] Create a performance baseline → See `docs/phase-1/performance-baseline.md`

## Phase 2: Types & Formatting

1. [x] Add mypy as a dependency
2. [x] Add ruff as a dependency
3. [x] Configure both mypy and ruff for this repo
4. [x] Set up the GitHub CI/CD pipeline to run both of those tools
5. [x] Resolve all mypy errors (61 → 0)

## Phase 3: Propose Architecture

1. [x] Research coding architectures → See `docs/phase-3/architecture-proposal.md`
2. [x] Weigh findings with project goals → Provider, Service, Repository patterns evaluated
3. [x] Include Pros & Cons → Each proposal section includes tradeoffs
4. [x] Create proposal document → `docs/phase-3/architecture-proposal.md`
5. [ ] Sign off on proposed architecture (AWAITING APPROVAL)
6. Note: This can be an iterative process

## Phase 4: Propose Configuration Architecture

1. [x] Find all entry points and config structures → 7 config sources documented
2. [x] Create markdown document with data structures → `docs/phase-4/configuration-proposal.md`
3. [x] Consider which values to expose vs hide → Tiered system proposed
4. [x] Suggest exposure via Stash Plugins vs code → Categories defined
5. [x] Document suggestions with pros/cons → Each section includes tradeoffs
6. [ ] Sign off on proposed config architecture (AWAITING APPROVAL)
7. Note: This can be an iterative process

## Phase 5: Consolidate Duplicates

1. [x] Double check architecture approval → Approved 2026-02-15
2. [x] Confirm architecture diagrams match → Verified
3. [x] Duplicate code consolidation → See `docs/phase-5/duplicate-consolidation.md`
4. [x] Describe duplicated code → escapeHtml (JS), formatDuration analysis
5. [x] Different purposes identified → formatDuration variants serve different needs (not duplicates)

## Phase 6: Restructure Modules

1. Perform the following for BOTH Python & JavaScript/CSS/HTML
2. Double check that the proposed architectures above are approved by me
3. Confirm that the architecture design and diagrams match with your description
4. Start with restructuring into modules
5. Describe which modules were created, deleted, modified, etc.

## Phase 7: Databases

1. [x] Research best practices → Embedding storage, SQLite optimization
2. [x] Document research → See `docs/phase-7/database-proposal.md`
3. [x] Understand databases in repo → 3 DBs (1 active, 2 stale, 3 empty)
4. [x] Suggest performance improvements → WAL mode, FAISS index, VACUUM
5. [x] Suggest consistency improvements → Backup system, integrity checks
6. ✅ NO MODIFICATIONS MADE TO DATA OR DATABASES (proposal only)

## Phase 8: Git Repo Organization

1. [x] Create a main branch and a dev branch → `main` exists, `dev` created 2026-02-15
2. [x] Make the dev branch the default → Done via `gh repo edit --default-branch dev`
3. [x] Add protections to the main branch → Requires GitHub Pro (private repo)
4. [x] Update CLAUDE.md → Development workflow + architecture diagram added

## User Preferences & Restrictions

**CRITICAL RESTRICTIONS:**
- DO NOT UNDER ANY CIRCUMSTANCES DELETE ANY OF THE DATABASES
- DO NOT UNDER ANY CIRCUMSTANCES DELETE ANY OF THE DATA FILES OR DIRECTORIES

**Allowed Actions:**
- Get database statistics and query databases to determine if they are still in use
- Get data file / directory level statistics to determine if they are still in use

**Database Considerations:**
- Redundancy (avoiding)
- Reliability
- Recoverability
- Robustness

**Code Organization Preferences:**
- Focus on creating atomic standalone files or larger more organized modules consisting of multiple atomic files
- Think of the atomicity of each unit of work or feature
- Prefer importing from many local modules/files over larger monolithic scripts
- Do not duplicate existing code
- Any duplicate code found should be consolidated into a "single source of truth"
- Main gripe is overlapping responsibilities - prefer single sources of truth
- Consider the frontend as important as the backend

## Final Deliverables

1. Rerun the performance metrics and report any differences
2. Create a mermaid diagram of the repo AFTER the clean up
3. Summarize any changes or design decisions made
4. Update CLAUDE.md:
   - Remove sections that are no longer relevant
   - Add the NEW architecture diagram to the top (with a datetime stamp)
   - Save the mermaid diagram in an external directory
   - Add a directive requiring updates to architecture diagram for structural changes
   - Add a block to the TOP that imports and displays the LATEST architecture diagram
   - Evaluate and update design principles
5. Review the original features table and ensure ALL features are still functioning
6. Evaluate the current test suite and find gaps in coverage, branches not tested, code never executed
7. Describe the current test suite in a separate markdown file. Suggest a path for TDD
8. Find future improvements to structure and architecture and describe them
9. Create a table of all files created during this process
10. Create a new README.md with an updated project description
11. Create a final report with datetime stamp

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 | **Complete** | Type checking (0 errors), tests (247 passed, 29% coverage), config/perf documented |
| Phase 1 | **Complete** | Architecture, database schemas, features, DESIGN.md, performance baseline |
| Phase 2 | **Complete** | mypy (0 errors), ruff configured, CI/CD set up |
| Phase 3 | **Approved** | Architecture proposal approved 2026-02-15 |
| Phase 4 | **Approved** | Configuration proposal approved 2026-02-15 |
| Phase 5 | **Complete** | Dead code removed (~4,100 lines), duplicates consolidated, tests pass |
| Phase 6 | **Complete** | Exception hierarchy + typed config module created |
| Phase 7 | **Complete** | Database proposal with recommendations (no changes made) |
| Phase 8 | **Complete** | dev branch created, set as default, CLAUDE.md updated with architecture |

## Documentation Files Created

| Phase | File | Description |
|-------|------|-------------|
| 0 | `docs/phase-0-checklist.md` | Pre-cleanup checklist completion |
| 1 | `docs/phase-1/architecture.md` | System architecture with mermaid diagrams |
| 1 | `docs/phase-1/database-schemas.md` | Database schemas and statistics |
| 1 | `docs/phase-1/features.md` | Feature documentation and dependencies |
| 1 | `docs/phase-1/DESIGN.md` | Visual design system |
| 1 | `docs/phase-1/performance-baseline.md` | Performance metrics baseline |
| 3 | `docs/phase-3/architecture-proposal.md` | Architecture improvement proposals |
| 4 | `docs/phase-4/configuration-proposal.md` | Configuration architecture proposal |
| 7 | `docs/phase-7/database-proposal.md` | Database improvements proposal (no changes made) |
| 8 | `docs/phase-8/git-organization.md` | Git branch strategy and commands |
| 5 | `docs/phase-5/duplicate-consolidation.md` | Duplicate code analysis and consolidation |
| 6 | `docs/phase-6/module-restructure-plan.md` | Module restructuring implementation plan |
