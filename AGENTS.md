# MoE training study

Read README.md and provenance/README.md first. Keep the completed experimental claims tied to recorded measurements. Distinguish character CE from subword CE, selected weights from FLOPs, and expert traffic from task-gradient evidence.

The repository consolidates current runnable code, four completed evidence packages, routing_followup/, and development_archive/. Preserve distinct historical snapshots and negative results. Never alter measured numeric values. Source and presentation sanitization is recorded separately from original source validation.

Checkpoints remain in an external backup and must stay out of Git. Other submission artifacts, including datasets and results, are tracked. Do not claim new training or checkpoint verification without performing it. Detach shared links before editing runnable source so historical copies remain unchanged. No remote publishing or agent delegation without explicit user authorization.

Validation: python3 -m unittest discover -s tests -v; python3 scripts/audit_colab_evidence.py; python3 -m scripts.audit_routing_followup. Run the follow-up tests from routing_followup/. Full checkpoint review requires --checkpoint-root pointing to the external backup.
