#!/bin/bash
export GIT_DIR="$(cd "$(dirname "$0")" && pwd)/.vp_git"
export GIT_WORK_TREE="$(cd "$(dirname "$0")" && pwd)"
git "$@"
