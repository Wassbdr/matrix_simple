# Security policy

## Supported versions
Only the `main` branch is supported.

## Reporting a vulnerability
Email **wassim.badraoui@epita.fr** with the subject `[SECURITY] gnn-rca`.
Do not open a public issue for security reports. Expect a first response within 7 days.

## Known sandbox-only credentials
The Kubernetes manifests under `k8s/app/` (`db.yaml`, `backend.yaml`) contain
the placeholder password `mypassword`. These manifests are intended for an
**isolated sandbox cluster** used only to generate chaos-injection datasets.
**Do not deploy them to a shared or production cluster.** Replace the password
with a Kubernetes Secret before any non-sandbox use.
