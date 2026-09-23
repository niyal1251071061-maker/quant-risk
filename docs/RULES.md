# Development Rules

## General
- Use Python 3.11+.
- Reuse existing functions. Do not duplicate logic.
- Keep functions small and focused.
- Do not modify unrelated files.

## Before Coding
- Read PRD.md and ARCHITECTURE.md.
- Make a plan for complex mathematical derivations.
- Verify gradient calculations numerically before training.

## Machine Learning Rules
- **No Data Leakage:** Strictly enforce chronological walk-forward validation.
- **Stationarity:** Always convert raw prices to log returns.
- **Custom Loss:** Derive the 1st (gradient) and 2nd (Hessian) derivatives analytically before implementing.
- **Baseline Comparison:** Always benchmark custom loss against MSE and Huber.

## UI
- Follow DESIGN.md.
- Include loading and error states.
- Display risk warnings clearly.

## Security
- Never expose API keys in code.
- Validate all ticker inputs using Pydantic in FastAPI.

## Git
- Make small, descriptive commits.
- Use `feat:`, `fix:`, `docs:` prefixes.