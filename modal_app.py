"""Run a LongProc config on a Modal GPU (one NVIDIA L4).

  modal run modal_app.py --config configs/longproc-0.5k-test-r0.yaml --stage prepare
  modal run modal_app.py --config configs/longproc-0.5k-test-r0.yaml --stage run --max-cases 48 --resume

`prepare` downloads the pinned model revision into a volume (needs a Modal secret named `huggingface`). `run` decodes
every case under the config's conditions and appends records to a results volume, in a directory named by the
config's hash; `scripts/run_shards.sh` repeats it until every page is done.
"""
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

app = modal.App('progress-aware-kv')
weights = modal.Volume.from_name('progress-aware-kv-models', create_if_missing=True)
results = modal.Volume.from_name('progress-aware-kv-runs', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', index_url='https://download.pytorch.org/whl/cu124')
         .pip_install_from_requirements(str(ROOT / 'requirements.remote.lock.txt'))
         .env({'HF_HOME': '/models/hf', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir(str(ROOT / 'src/orderkv'), remote_path='/root/orderkv'))


def manifest_path(model_id: str) -> Path:
    return Path('/models/model_manifest-' + model_id.replace('/', '--') + '.json')


@app.function(image=image, cpu=(1.0, 2.0), memory=(2048, 4096), volumes={'/models': weights}, timeout=1800,
              secrets=[modal.Secret.from_name('huggingface')])
def prepare_model(revision: str, model_id: str):
    from huggingface_hub import HfApi, snapshot_download
    path = manifest_path(model_id)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['revision'] != revision:
            raise ValueError('Model volume holds a different revision')
        return existing
    if HfApi().model_info(model_id, revision=revision).sha != revision:
        raise ValueError('Expected an immutable model commit')
    model_path = snapshot_download(model_id, revision=revision,
                                   allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    manifest = {'model_id': model_id, 'revision': revision, 'path': model_path}
    path.write_text(json.dumps(manifest, indent=2))
    weights.commit()
    return manifest


@app.function(image=image, gpu='L4', cpu=(1.0, 2.0), memory=(16384, 24576),
              volumes={'/models': weights, '/results': results}, timeout=1800, retries=0)
def run(config_values: dict, public_jsonl: str, private_jsonl: str, baseline_budget_json: str,
        resume: bool, max_cases: int, max_seconds: int):
    from dataclasses import replace
    import tempfile
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orderkv.config import Config
    from orderkv.cursor import run_cursor_study
    started = time.perf_counter()
    config = Config(**config_values).validate()
    manifest = json.loads(manifest_path(config.model_id).read_text())
    if manifest['revision'] != config.model_revision:
        raise ValueError('Prepared model and config revision differ')
    torch.manual_seed(config.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Loading straight to the GPU keeps a 7B/8B model inside the container's host memory.
    model = AutoModelForCausalLM.from_pretrained(manifest['path'], device_map={'': 'cuda'}, local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                                                 attn_implementation=config.attention_backend, low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(manifest['path'], local_files_only=True, trust_remote_code=False)
    series = Path('/results') / config.digest()
    series.mkdir(parents=True, exist_ok=True)
    (series / 'config.json').write_text(json.dumps(config_values, indent=2))
    with tempfile.TemporaryDirectory() as directory:
        split = Path(config.dataset).stem.removesuffix('_public')
        runtime = replace(config, dataset=f'{directory}/{split}_public.jsonl',
                          private_labels=f'{directory}/{split}_private.jsonl')
        Path(runtime.dataset).write_text(public_jsonl)
        Path(runtime.private_labels).write_text(private_jsonl)
        if config.baseline_budget_path:
            runtime = replace(runtime, baseline_budget_path=f'{directory}/baseline_budget.json')
            Path(runtime.baseline_budget_path).write_text(baseline_budget_json)
        records = run_cursor_study(runtime, model, tokenizer, series, resume=resume, max_cases=max_cases,
                                   max_seconds=max(0, max_seconds - (time.perf_counter() - started)),
                                   checkpoint=results.commit, identity_config=config)
    results.commit()
    return {'result_path': str(series), 'statuses': [r['status'] for r in records]}


@app.local_entrypoint()
def main(config: str, stage: str = 'run', max_cases: int = 48, max_seconds: int = 1650, resume: bool = False):
    from dataclasses import asdict
    from orderkv.config import load_config
    values = load_config(config)
    if stage == 'prepare':
        print(prepare_model.remote(values.model_revision, values.model_id))
    elif stage == 'run':
        budget = Path(values.baseline_budget_path).read_text() if values.baseline_budget_path else ''
        print(run.remote(asdict(values), Path(values.dataset).read_text(), Path(values.private_labels).read_text(),
                         budget, resume, max_cases, max_seconds))
    else:
        raise ValueError('stage must be prepare or run')
