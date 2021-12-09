#!/bin/bash

set -x
################################################################################
# File:    build_docs.sh
# Purpose: Build documentation with sphinx and update github
#          pages. Github actions executes this script via
#            .github/workflows/docs_pages_workflow.yml
#
# Authors: Mike Schmidt <mikeschmidt@schmidtgracen.com>
#          forked from Michael Altfield's https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1/
# Created: 2021-12-08
# Updated: 2021-12-08
# Version: 0.0.1
################################################################################

################################################################################
# 
################################################################################

apt-get update
apt-get install -y git rsync python3-sphinx python3-sphinx-rtd-theme

################################################################################
#
################################################################################

pwd
ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

################################################################################
#
################################################################################

# Build documentation
make -C docs clean
make -C docs html

################################################################################
#
################################################################################

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

docroot=`mktemp -d`
rsync -av "docs/_build/html/" "${docroot}/"

pushd "${docroot}"

# This is generated freshly each time, deleting any prior history.
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# This file makes paths with underscores OK
touch .nojekyll

echo "
This branch is a temporary cache of documentation. For actual documentation,
see the main branch's docs folder.

Thanks to Michael Altfield for his instructions in how to do this.
https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
">README.md

ls -la .
git add .
git status
git commit -am "updating docs for commit ${GITHUB_SHA} on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"

git push deploy gh-pages --force

popd

exit 0

