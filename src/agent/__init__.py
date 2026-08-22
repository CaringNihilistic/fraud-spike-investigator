"""P2: policy-gated LLM investigator.

The agent EXPLAINS a spike. It never decides. Every action it recommends is
re-validated by src/policy/engine.validate_recommendation() before it can
reach anything, and out-of-allowlist recommendations degrade to REVIEW.
"""
