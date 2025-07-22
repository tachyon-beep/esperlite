# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of the Esper platform seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: security@esper-platform.org

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting)
- Full paths of source file(s) related to the issue
- Location of affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue

## Preferred Languages

We prefer all communications to be in English.

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

## Security Best Practices

When deploying Esper:

1. **API Keys**: Always use strong, randomly generated API keys
2. **Network Security**: Use TLS for all external communications
3. **Secrets Management**: Never commit secrets to version control
4. **Input Validation**: The platform validates all kernel inputs
5. **Isolation**: Run services in isolated containers
6. **Updates**: Keep all dependencies up to date

## Known Security Considerations

### Kernel Execution
- Kernels are validated before execution
- Execution happens in isolated environments
- Resource limits are enforced

### Data Privacy
- Training data remains local
- No telemetry is sent without explicit configuration
- Model artifacts are stored securely

### Network Security
- All services support TLS
- Authentication required for all API endpoints
- Rate limiting prevents abuse