"""
Patch massive-topology-N50.yaml to run the dummy service from a ConfigMap
on the public python:3.9-slim image, removing the dependency on a custom
`rca-dummy-service:latest` image that is not available on the cluster.

Produces:
    rca_pipeline/k8s_massive/massive-topology-N50-configmap.yaml

Usage:
    python -m rca_pipeline.k8s_massive.patch_topology_for_configmap
    kubectl apply -f rca_pipeline/k8s_massive/dummy_service_configmap.yaml
    kubectl apply -f rca_pipeline/k8s_massive/massive-topology-N50-configmap.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).parent
INPUT_YAML = HERE / "massive-topology-N50.yaml"
OUTPUT_YAML = HERE / "massive-topology-N50-configmap.yaml"
CONFIGMAP_YAML = HERE / "dummy_service_configmap.yaml"
DUMMY_SCRIPT = HERE / "dummy_service.py"


def build_configmap() -> str:
    script_src = DUMMY_SCRIPT.read_text()
    indented = "\n".join("    " + line for line in script_src.splitlines())
    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: dummy-service-script
  namespace: rca-sandbox
data:
  main.py: |
{indented}
  requirements.txt: |
    fastapi==0.115.0
    uvicorn==0.30.6
    requests==2.32.3
    prometheus-fastapi-instrumentator==7.0.0
"""


def patch_yaml(src: str) -> str:
    """Replace image + add command/volumeMounts/volumes for each Deployment.

    Also injects Prometheus scrape annotations on the pod template so the
    cluster's auto-discovery scrape config (annotation-based) picks up
    every pod's /metrics endpoint exposed by prometheus_fastapi_instrumentator.

    The original block looks like:
        - name: <svcname>
          image: rca-dummy-service:latest
          imagePullPolicy: IfNotPresent
          ports:
          - containerPort: 8000
          env:
          - name: SERVICE_NAME
            value: <svcname>
          - name: UPSTREAMS
            value: "..."
          resources:
            ...

    We rewrite to use python:3.9-slim + pip install + run /app/main.py.
    """
    # Inject prometheus annotations into every pod template metadata block.
    # The pattern `    metadata:\n      labels:` appears once per Deployment
    # at the pod template level.
    src = src.replace(
        "  template:\n    metadata:\n      labels:",
        "  template:\n    metadata:\n      annotations:\n"
        "        prometheus.io/scrape: \"true\"\n"
        "        prometheus.io/port: \"8000\"\n"
        "        prometheus.io/path: \"/metrics\"\n"
        "      labels:",
    )
    out_lines = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        out_lines.append(line)
        if line.strip().startswith("- name:") and i + 1 < len(lines) and "image: rca-dummy-service" in lines[i + 1]:
            # We are on the container's `- name: <svcname>` line.
            # Skip the original image + imagePullPolicy lines (i+1, i+2)
            # then keep ports/env, then insert command + volumeMounts before resources.
            out_lines.append("        image: python:3.9-slim")
            out_lines.append("        imagePullPolicy: IfNotPresent")
            out_lines.append("        command: [\"/bin/sh\", \"-c\"]")
            out_lines.append("        args:")
            out_lines.append("          - |")
            out_lines.append("            set -e")
            out_lines.append("            pip install --no-cache-dir -q -r /app/requirements.txt")
            out_lines.append("            exec python /app/main.py")
            # Skip the original image + imagePullPolicy lines we just replaced
            i += 3  # original `- name:` then `image:` then `imagePullPolicy:`
            # Copy lines until we hit `resources:` (insert volumeMount before)
            while i < len(lines) and not lines[i].strip().startswith("resources:"):
                out_lines.append(lines[i])
                i += 1
            # Insert volumeMount lines before resources
            out_lines.append("        volumeMounts:")
            out_lines.append("        - name: dummy-script")
            out_lines.append("          mountPath: /app")
            # Now append resources block + everything until the volumes-relevant area
            while i < len(lines) and not (lines[i].strip().startswith("---") or lines[i].rstrip() == "      restartPolicy: Always"):
                # Detect end of containers section to insert volumes
                # Heuristic: insert volumes right before the next top-level `---` separator,
                # but inside the same template.spec
                out_lines.append(lines[i])
                i += 1
            # Insert pod-level volumes before document separator
            out_lines.append("      volumes:")
            out_lines.append("      - name: dummy-script")
            out_lines.append("        configMap:")
            out_lines.append("          name: dummy-service-script")
            # Continue with original document separator
            continue
        i += 1
    return "\n".join(out_lines) + "\n"


def main():
    if not INPUT_YAML.exists():
        raise SystemExit(f"missing {INPUT_YAML}")
    src = INPUT_YAML.read_text()
    patched = patch_yaml(src)

    CONFIGMAP_YAML.write_text(build_configmap())
    OUTPUT_YAML.write_text(patched)

    print(f"Wrote {CONFIGMAP_YAML} ({os.path.getsize(CONFIGMAP_YAML)} bytes)")
    print(f"Wrote {OUTPUT_YAML} ({os.path.getsize(OUTPUT_YAML)} bytes)")
    print()
    print("Apply with:")
    print(f"  kubectl apply -f {CONFIGMAP_YAML}")
    print(f"  kubectl apply -f {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
