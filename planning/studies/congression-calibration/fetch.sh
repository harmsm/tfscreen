#!/usr/bin/env

rsync -av --exclude '*.h5' -e ssh $(T):${1} .
