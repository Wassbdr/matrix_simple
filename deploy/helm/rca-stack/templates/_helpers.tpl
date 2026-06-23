{{/* Full image ref for a given component name (serve/watcher/train). */}}
{{- define "rca.image" -}}
{{- $root := index . 0 -}}
{{- $name := index . 1 -}}
{{- printf "%s/%s/rca-sta-%s:%s" $root.Values.image.registry $root.Values.image.owner $name $root.Values.image.tag -}}
{{- end -}}

{{/* initContainer that fetches the bundle from the registry by alias. */}}
{{- define "rca.fetchBundleInit" -}}
{{- $root := index . 0 -}}
{{- $alias := index . 1 -}}
- name: fetch-bundle
  image: {{ include "rca.image" (list $root "train") }}
  imagePullPolicy: {{ $root.Values.image.pullPolicy }}
  command:
    - python
    - -m
    - serving.fetch_bundle
    - --model={{ $root.Values.model.registry.name }}
    - --alias={{ $alias }}
    - --dest={{ $root.Values.model.bundlePath }}
  env:
    - name: MLFLOW_TRACKING_URI
      value: {{ $root.Values.mlflow.trackingUri | quote }}
    {{- if $root.Values.mlflow.authSecret }}
    - name: MLFLOW_TRACKING_USERNAME
      valueFrom:
        secretKeyRef: { name: {{ $root.Values.mlflow.authSecret }}, key: MLFLOW_TRACKING_USERNAME, optional: true }
    - name: MLFLOW_TRACKING_PASSWORD
      valueFrom:
        secretKeyRef: { name: {{ $root.Values.mlflow.authSecret }}, key: MLFLOW_TRACKING_PASSWORD, optional: true }
    {{- end }}
  volumeMounts:
    - name: model-bundle
      mountPath: {{ $root.Values.model.bundlePath }}
{{- end -}}
