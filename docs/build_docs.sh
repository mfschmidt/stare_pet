#!/bin/bash

set -x
################################################################################
# File:    build_docs.sh
# Purpose: Build documentation with sphinx and update github
#          pages. Github actions executes this script via
#            .github/workflows/docs_pages_workflow.yml
#
# Authors: Mike Schmidt <mikeschmidt@schmidtgracen.com>
#          forked from Michael Altfield https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1/
# Created: 2021-12-08
# Updated: 2021-12-08
# Version: 0.0.1
################################################################################

################################################################################
# Configure the VM environment
################################################################################

apt-get update
apt-get install -y git rsync \
                   python3-sphinx python3-sphinx-rtd-theme \
                   python3-sphinx-argparse \
                   python3-numpy python3-pandas python3-sklearn python3-humanize

################################################################################
# Remember the context
################################################################################

pwd
ls -lah
SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
export SOURCE_DATE_EPOCH

################################################################################
# Build the documentation
################################################################################

make -C docs clean
make -C docs html

################################################################################
# Update ephemeral github pages branch
################################################################################

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

doc_root=$(mktemp -d)
rsync -av "docs/_build/html/" "${doc_root}/"

pushd "${doc_root}" || exit

# This is generated freshly each time, deleting any prior history.
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# This file makes paths with underscores OK
touch .nojekyll

echo "
This branch is a temporary cache of bot-generated documentation.
For actual documentation, see the main branch's docs folder.

Thanks to Michael Altfield for his instructions in how to do this.
https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
">README.md

ls -la .
git add .
git status
DATE_STRING=$(date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds)
git commit -am "updating docs for commit ${GITHUB_SHA} on ${DATE_STRING} from ${GITHUB_REF} by ${GITHUB_ACTOR}"

git push deploy gh-pages --force

popd || exit

exit 0

