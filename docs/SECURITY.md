# QuantRisk AI — Security and Data Protection

## Trust boundaries

The ML scripts and report files are local research inputs. SQLite is the local
serving store. FastAPI exposes read-only derived results. Streamlit is the user
interface. Gemini is an optional external service and receives only selected
financial-news headlines.

## Validation and authorization

- Tickers are allow-listed to `NIFTY50` and `SP500`.
- API responses use strict Pydantic schemas.
- SQL queries use fixed table names derived from the allow-list; user input is
  never interpolated without validation.
- The current local API is read-only and does not require JWT authentication.
- A production deployment must restrict CORS origins, add authentication,
  rate limiting, TLS, and audit logging before public exposure.

## Secrets

Store `GEMINI_API_KEY` only in the environment or deployment secret manager.
Never commit it to source, reports, SQLite, logs, or screenshots. The
dashboard falls back to a neutral status when the key is absent.

## Financial safety

This platform provides analytical decision support only. It does not execute
trades or provide financial advice. Leakage checks, walk-forward validation,
transaction-cost analysis, and corrected log-return metrics must be rerun when
data or model configurations change.

## Operational controls

- Regenerate SQLite after report changes.
- Keep backups of the database outside the repository if it contains sensitive
  research output.
- Do not expose debug reload servers in production.
- Review dependency updates for vulnerabilities before deployment.
