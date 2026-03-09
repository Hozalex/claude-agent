---
description: Triage a monitoring alert. Use when the user provides an alert from Alertmanager, Grafana, Prometheus, or any other monitoring system and asks to check, investigate, or explain it.
agent: sonnet
---

# Alert Triage

Delegate to the `sonnet` subagent. Analyze the alert and run targeted kubectl checks.

## Steps

1. **Parse the alert**: extract alertname, severity, affected resource (pod/node/namespace), labels, and annotations.

2. **Check affected resources**:
```bash
# If alert mentions a pod
kubectl get pod <pod> -n <namespace> -o wide
kubectl describe pod <pod> -n <namespace>
kubectl logs <pod> -n <namespace> --tail=50 --previous 2>/dev/null || true

# If alert mentions a node
kubectl describe node <node>
kubectl get pods -A --field-selector spec.nodeName=<node>

# General resource state
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | tail -20
```

3. **Assess severity**:
   - Is the alert actively firing or already resolved?
   - What is the blast radius (how many users/services affected)?
   - Is there a related incident already in progress?

4. **Check cluster-wide health** if the alert is critical:
```bash
kubectl get nodes
kubectl get pods -A | grep -E 'CrashLoopBackOff|Error|OOMKilled|Evicted'
```

## Response format

Line 1: 🔴 CRITICAL / 🟠 WARNING / 🟡 DEGRADED / ✅ OK — one-line summary.
Then:
- **Root cause**: what is actually wrong
- **Affected**: list of impacted resources
- **Action**: specific command(s) the engineer should run (read-only suggestions or write commands they must approve)

Keep under 25 lines. Plain text, no markdown headers.
