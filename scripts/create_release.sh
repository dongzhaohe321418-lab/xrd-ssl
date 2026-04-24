#!/bin/bash
# Create GitHub release for XRD-SSL
# Run after final review and before arXiv submission

set -e

echo "Creating GitHub release..."

# Tag the release
git tag -a v1.0 -m "Preprint v1 — Sort-Match SSL for XRD Peak Prediction"

# Push tag
git push origin v1.0

# Create release
gh release create v1.0 \
    --title "Preprint v1 — XRD-SSL" \
    --notes-file FINAL_REPORT.md

echo "Release created: https://github.com/$(gh repo view --json owner,name -q '.owner.login + \"/\" + .name')/releases/tag/v1.0"
