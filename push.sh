#!/bin/bash
# Quick commit and push to ro-dev-1
# Usage: ./push.sh [optional commit message]

MSG=${1:-"update $(date '+%Y-%m-%d %H:%M:%S')"}

git add -A
git commit -m "$MSG"
git push origin ro-dev-1