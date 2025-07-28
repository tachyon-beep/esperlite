---
# Esperlite Project Context

## Project Overview
Esperlite is a morphogenetic neural network system that enables runtime adaptation of models through dynamic kernel loading. The system has completed Phase 4 implementation with full service integration.

## Current Status (Phase 4 Complete)
- ✅ Core morphogenetic architecture implemented
- ✅ Service mesh integrated (Tolaria, Tamiyo, Nissa)
- ✅ Message bus operational (Redis Streams)
- ✅ All 137 test failures resolved
- ✅ Test suite peer reviewed and improved

## Key Technical Context
- **Python Version**: 3.12 (migrated from aioredis to redis.asyncio)
- **Architecture**: Async-first with sync fallback
- **Testing**: Real service infrastructure with minimal mocking
- **Services**: Tolaria (training), Tamiyo (adaptation), Nissa (observability)

## Known Limitations
- Sync kernel execution only supports fallback (no actual kernel execution)
- Phase 4 message bus tests have JSON serialization issues (not in original scope)

## Development Guidelines
- Prefer integration tests over unit tests for complex systems
- Use real services when available, mock only external dependencies
- Test meaningful behavior, not implementation details
- Maintain async-first design with graceful degradation

---
# Codacy Rules
Configuration for AI behavior when interacting with Codacy's MCP Server

## After ANY successful file edit
- YOU MUST run the `codacy_cli_analyze` tool for each edited file
- If any issues are found, propose and apply fixes immediately
- This is a critical requirement, not optional

## After ANY dependency installation
- IMMEDIATELY run `codacy_cli_analyze` with tool set to "trivy"
- Check for security vulnerabilities in new packages
- Stop all operations if vulnerabilities found
- Fix security issues before continuing

## Repository setup
- Use `git remote -v` to determine provider/organization/repository
- Provider mapping: GitHub = "gh", Bitbucket = "bb", GitLab = "gl"
- If repository not found in Codacy, offer to run `codacy_setup_repository`
---
