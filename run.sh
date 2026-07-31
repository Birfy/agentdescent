#!/bin/zsh
source ~/.zshrc >/dev/null 2>&1
cd /tmp/ad-adas
export PYTHONPATH=.
{ time python3 -u -m examples.adas_meta_agent_search --provider openai --model deepseek-v4-flash \
    --hard --langs bn,sw,te,th --per-lang 150 --generations 3 --yes ; } > /tmp/ad-adas/adas.txt 2>&1
echo DONE >> /tmp/ad-adas/status
