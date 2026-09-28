#!/bin/zsh
# Run a config shard by shard on Modal (each shard resumes the last) until TARGET pages have every condition, then
# leave the full records in outputs/. Usage: scripts/run_shards.sh CONFIG CASES [MAX_SHARDS] [TARGET]
cd "$(dirname "$0")/.."
config=$1; cases=$2; shards=${3:-12}; target=${4:-$2}
mkdir -p logs outputs
series=$(PYTHONPATH=src .venv/bin/python -m orderkv.experiment --config $config | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['config_hash'])")
for i in $(seq 1 $shards); do
  log=logs/shard-$(basename $config .yaml)-$(date +%s).log
  .venv/bin/modal run --detach modal_app.py --config $config --stage run --max-cases $cases --resume > $log 2>&1 &
  pid=$!
  # Modal occasionally cancels the remote call while the local client hangs; no shard outlives the 1800 s timeout.
  waited=0
  while kill -0 $pid 2>/dev/null && [ $waited -lt 2100 ]; do sleep 10; waited=$((waited + 10)); done
  if kill -0 $pid 2>/dev/null; then echo "watchdog: client hung, killing"; kill $pid; sleep 2; fi
  wait $pid
  rc=$?
  rm -rf outputs/$series
  .venv/bin/modal volume get progress-aware-kv-runs $series outputs/ > /dev/null 2>&1
  done_pages=$(.venv/bin/python -c "
import json, yaml
try: rows = [json.loads(l) for l in open('outputs/$series/runs.jsonl')]
except FileNotFoundError: rows = []
wanted = yaml.safe_load(open('$config'))['conditions']
done = {(r['case_id'], r['condition']) for r in rows if r['status'] in ('complete', 'capped')}
print(sum(all((c, w) in done for w in wanted) for c in {c for c, _ in done}))")
  echo "shard $i rc=$rc pages done=$done_pages/$target"
  [ "$done_pages" -ge "$target" ] && exit 0
  if [ $rc -ne 0 ]; then
    failures=$((${failures:-0} + 1))
    [ $failures -ge 3 ] && { echo "stopping after 3 consecutive failures"; exit $rc; }
  else
    failures=0
  fi
done
