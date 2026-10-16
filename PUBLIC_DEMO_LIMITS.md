# Public demo limits

Ripple remains publicly accessible. The server applies these small demo limits:

- Source videos: at most 60 seconds, checked before cut detection and again before
  starting a saved job.
- Jobs: at most 600 estimated Runway credits ($6 at $0.01/credit), checked before
  starting any audio conversion or generation. This is an estimate, not a provider
  billing cap, and published workflow settings must agree with the estimator.
- Work: at most two preparation workers and two active generation jobs per server.
  Additional work receives a busy response rather than creating more workers.
- Submission requests: at most 30 per browser identity and 120 across the whole
  server per minute. Cancellation remains available when those limits are reached.
- Optional direct ElevenLabs transcription: disabled by default, including when an
  ElevenLabs key is configured. Explicitly set `RIPPLE_ENABLE_TRANSCRIPTION=true`
  to opt in to its separate billing. Runway preset voice conversion still works.

These limits apply on the server, including requests made outside the interface.
The request counters and active-worker caps are per process and reset on restart.
Browser identity is replaceable; the global limit still applies when it changes.
Existing upload byte limits, job ownership checks, cancellation, and the disabled
paid retry/regeneration behavior also remain in place.

The operator keeps Runway automatic refill disabled. Depleted Runway credits remain
the final spending backstop. This demo does not implement verified user quotas,
distributed budgets, or protection against all forms of resource exhaustion.
