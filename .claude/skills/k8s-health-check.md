---
description: Check Kubernetes cluster health. Use when the user asks to check the cluster, look for problems, find failing or pending pods, check node status, or diagnose any Kubernetes issue.
agent: haiku
---

# Kubernetes Cluster Health Check

Delegate to the `haiku` subagent. Run these checks:

### 1. Node status
```bash
kubectl get nodes -o wide
```
Flag: NotReady, SchedulingDisabled, MemoryPressure, DiskPressure, PIDPressure.

### 2. Problem pods (all namespaces)
```bash
kubectl get pods -A --field-selector=status.phase!=Running,status.phase!=Succeeded
kubectl get pods -A | grep -E 'CrashLoopBackOff|Error|OOMKilled|Evicted|Pending|ImagePullBackOff'
```

### 3. Recent warning events
```bash
kubectl get events -A --field-selector=type=Warning --sort-by='.lastTimestamp' | tail -20
```

### 4. Node resource usage
```bash
kubectl top nodes 2>/dev/null || echo "metrics-server not available"
```

## Response format

First line: "✅ Cluster OK" or "⚠️ Issues found: N".
Then list only problems: node/pod name, namespace, state, restart count.
Skip healthy resources. Keep it under 20 lines.
