#!/bin/sh

set -e

while [ ! -f ./stop ]; do
	AIS=`shuf -e 'kb.stei_adt' 'kb.stei_at' 'dt.ste' 'dt.stei' | tr "\n" " "`
	echo "$AIS"
	#rm logs/*
	python3 scripts/dicewars-ai-only.py --ai $AIS #-l ./logs/
done
rm ./stop
