#!/usr/bin/env bash

# egrep -rv "^\s*#" {agent,network,app,main,target}.py | wc -l
egrep -rv "^\s*#" `find . -type f -name "*.py"` | wc -l
