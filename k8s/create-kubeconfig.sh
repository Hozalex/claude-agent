#!/usr/bin/env bash
# Creates sre-bot-kubeconfig.yaml for local Docker usage.
# Usage: ./k8s/create-kubeconfig.sh
set -euo pipefail

NAMESPACE="default"
SECRET_NAME="sre-bot-token"
OUTPUT="sre-bot-kubeconfig.yaml"

echo "→ Applying RBAC..." >&2
kubectl apply -f "$(dirname "$0")/rbac.yaml"

echo "→ Waiting for token..." >&2
for i in $(seq 1 15); do
  TOKEN=$(kubectl get secret "${SECRET_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.data.token}' 2>/dev/null | base64 --decode || true)
  [ -n "${TOKEN}" ] && break
  sleep 2
done

[ -z "${TOKEN}" ] && { echo "ERROR: token not populated" >&2; exit 1; }

CLUSTER_NAME=$(kubectl config view --minify -o jsonpath='{.clusters[0].name}')
CLUSTER_SERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')
CLUSTER_CA=$(kubectl config view --minify --raw \
  -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')

cat > "${OUTPUT}" <<EOF
apiVersion: v1
kind: Config
clusters:
  - name: ${CLUSTER_NAME}
    cluster:
      server: ${CLUSTER_SERVER}
      certificate-authority-data: ${CLUSTER_CA}
contexts:
  - name: sre-bot
    context:
      cluster: ${CLUSTER_NAME}
      user: sre-bot
current-context: sre-bot
users:
  - name: sre-bot
    user:
      token: ${TOKEN}
EOF

echo "✅ ${OUTPUT} created. Test with:"
echo "   kubectl --kubeconfig=${OUTPUT} get pods -A"
