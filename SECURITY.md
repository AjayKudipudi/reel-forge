# Security Policy

## Reporting a Vulnerability

If you discover a security issue (leaked credentials, remote-code-execution
vector in the pipeline, AWS escalation path, etc.), **do not open a public
GitHub issue**. Instead, email the maintainer directly:

**venkatajay903@gmail.com**

Include:
- Description of the issue
- Steps to reproduce (if applicable)
- Affected file(s) / commit(s)
- Impact (data exposure, account compromise, etc.)

You can expect an initial response within 7 days. We will work with you
to validate the issue, prepare a fix, and coordinate disclosure.

## Common security pitfalls in this project

This project handles AWS credentials, Hugging Face tokens, and S3 storage.
When contributing:

- **Never commit `.env`** — `.gitignore` excludes it but verify before pushing.
- **Never paste tokens in commit messages, PR descriptions, or issue bodies.**
- **Rotate tokens** if you accidentally expose them in logs or screenshots.
- The cloud-init template (`reel_forge/ec2/launch.py`) embeds the HF token
  in user-data sent to EC2. Treat it as sensitive.
- IAM permissions should follow least-privilege — see
  [`docs/05_aws_setup.md`](docs/05_aws_setup.md) for the documented minimum set.

## Supported versions

Only the latest release on `main` receives security fixes. Older versions
may have unfixed issues.
