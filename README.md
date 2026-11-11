# openset_query

A warm open-vocabulary object query server for robot maps. It keeps
[SAM 3](https://github.com/facebookresearch/sam3) and a SigLIP2 image-text model
resident on the GPU and answers text queries ("a fire extinguisher") over the
keyframes of a [Hydra](https://github.com/MIT-SPARK/Hydra) image store and the
scene graph in Neo4j (via [heracles](https://github.com/GoldenZephyr/heracles)):

1. rank the keyframes against the prompt with SigLIP2,
2. run SAM 3 on the best frames (frustum-aware early stop, cached image features),
3. reproject the masks with depth into 3D and cluster them into objects,
4. return candidates (dry run) or write the objects, their observations and masks
   into Neo4j and the image store.

It is the backend of the object search in the ADT4 web console
([adt4_console](https://github.com/harelb/adt4_console)), which talks to it over
HTTP (default port 8077). It runs on the base station only.

## Install

The server runs in its own venv (SAM 3 pins `numpy<2`). `setup_env.sh` is
non-interactive and idempotent; it reuses an existing venv.

```bash
export ADT4_WS=...   # colcon workspace with src/awesome_dcist_t4
export ADT4_ENV=...  # where the ADT4 python environments live
bash setup_env.sh            # venv at ${SAM3_ENV:-$ADT4_ENV/sam3}
bash setup_env.sh --weights  # model weights into the Hugging Face cache
bash setup_env.sh --check    # report venv, torch/CUDA, imports, weights
```

It installs torch/torchvision 2.12/0.27 (CUDA 13.0 wheels; `TORCH_INDEX` to
change), SAM 3 from [harelb/sam3](https://github.com/harelb/sam3) at a pinned
commit (three compatibility commits on top of facebookresearch/sam3), `spark_dsg`
and `heracles` from the ADT4 workspace, Hydra's pure-python
`hydra_python.image_store` (as a path entry into the workspace's Hydra checkout;
the compiled Hydra bindings are not needed), and this package (editable).
`requirements-lock.txt` constrains third-party versions to the deployed
environment.

### Model weights

Weights are not part of this repository. The server loads them from the Hugging
Face cache:

| Hugging Face repo | files | used for | access |
| --- | --- | --- | --- |
| `facebook/sam3` | `config.json`, `sam3.pt` | SAM 3 image model | gated: accept the license on the model page |
| `google/siglip2-so400m-patch16-naflex` | config, tokenizer, `model.safetensors` | frame ranking, crop scoring | open |

`setup_env.sh --weights` downloads what is missing with `hf download`. When that
is not possible (no token, license not accepted, offline) it stops with
instructions, for example:

```bash
# a) accept the license at https://huggingface.co/facebook/sam3, then
$ADT4_ENV/sam3/bin/hf auth login
bash setup_env.sh --weights
# b) or copy the cache from a machine that has it
rsync -a <host>:.cache/huggingface/hub/models--facebook--sam3 ~/.cache/huggingface/hub/
```

Run the server with `HF_HUB_OFFLINE=1` once the weights are cached.

## Run

```bash
HF_HUB_OFFLINE=1 \
HERACLES_NEO4J_URI=bolt://localhost:7687 \
HERACLES_NEO4J_USERNAME=neo4j HERACLES_NEO4J_PASSWORD=<password> \
$ADT4_ENV/sam3/bin/openset-query-server --port 8077 --conf 0.25 --compile \
    --live-store-root $HOME/adt4_output/live
# or: $ADT4_ENV/sam3/bin/python -m openset_query.query_server ...
```

| variable / flag | meaning |
| --- | --- |
| `HERACLES_NEO4J_URI`, `HERACLES_NEO4J_USERNAME`, `HERACLES_NEO4J_PASSWORD` | scene graph database (keyframes, objects, observations); the URI defaults to `bolt://localhost:7687` and the user to `neo4j`; the password is required |
| `HF_HUB_OFFLINE=1` | use the cached weights only |
| `--port` | HTTP port (default 8077) |
| `--conf` | SAM 3 confidence threshold (default 0.5; the console deployment uses 0.25) |
| `--compile` | `torch.compile` the SAM 3 image model and warm it up at start |
| `--image-store DIR...` / `IMAGE_STORE` | image stores to search besides the ones the Neo4j map names |
| `--live-store-root DIR` / `SAM3_LIVE_STORE_ROOT` | also search live mirrors `<root>/<robot>/<epoch>/image_store` (re-scanned on `/reload`) |
| `--feature-cache-dir` / `SAM3_FEATURE_CACHE_DIR` | SAM 3 feature cache (default `<image_store>/cache/`) |
| `--admission-policy {off,margin,geometric}` / `SAM3_ADMISSION_POLICY` | grading of detector candidates (default `off`: writes stay candidates; a console commit admits them) |
| `--admission-calibration FILE` / `SAM3_ADMISSION_CALIBRATION` | calibration artifact for `margin`/`geometric`, produced by `python -m openset_query.eval.calibrate_admission` |

All flags: `openset-query-server --help`.
`python -m openset_query.precompute` precomputes the SigLIP2 embeddings and the
SAM 3 feature cache offline; `python -m openset_query.ingest_keyframes` writes
keyframe embeddings into Neo4j.

## HTTP API

All bodies are JSON. `openset_query.query_client` is a small stdlib client.

| route | request | response |
| --- | --- | --- |
| `GET /health` | | `status`, `keyframes`, `sources`, `pending`, `busy`, `conf_threshold`, `feature_cache`, ... |
| `POST /query` | `prompt`, optional `mode`, `top_k`, `max_frames`, `patience`, `min_ts`, `source`, `conf`, `dry_run`, `render`, `association` | detected objects; with `dry_run` a `pending_id` and `candidates` (views with mask RLE and annotated images) |
| `POST /commit` | `pending_id`, `index` | writes one dry-run candidate (admitted): `symbol`, `created`, `merged`, `observation_ids`, `state_version`, `map_version` |
| `POST /discard` | `pending_id` | `{"discarded": bool}` |
| `POST /rank` | `prompt`, `top_k`, `source`, `min_ts`, `thumbs` | SigLIP2 frame ranking only (no SAM 3, nothing written) |
| `GET /frame` | `key=<frame_key>` or `pending_id=&index=[&view=]`, optional `max_dim`, `mask`, `bbox` | a keyframe as a JPEG data URI, optionally annotated |
| `POST /reload` | | re-read the keyframe pool (Neo4j + live mirrors) |
| `POST /admission` | `policy`, optional `calibration` | switch the admission policy without reloading models |

The full contract is in the docstring of `openset_query/query_server.py`.

## Requirements on the ADT4 workspace

- `hydra_python.image_store` from Hydra `develop` (2ae75e8 or later).
- `heracles` master ([robustrobotics/heracles](https://github.com/robustrobotics/heracles)):
  `Neo4jWrapper`, `constants`, `str_to_ns_value`, `node_to_dict`,
  `insert_nodes_to_db`, `insert_edges` and `link_observations_to_frames`. That is
  all querying and committing need; the `Observation` writer and the map version
  bump live in `openset_query.heracles_compat`, with the same node shapes heracles
  writes.
- Image stores in the `hydra_python.image_store` format. Stores written by the
  earlier ADT4 image-store fork that still hold sub-keyframes (frame kind 1)
  must be converted first.

The map version is bumped after a commit with the newest API that is importable:
`heracles_ros.map_state.notify_map_change` (recording the changed layer and
object when it accepts `changed_layers` / `changed_ids`), else
`heracles.map_reload.bump_map_version` with the change, else the same Cypher
(`_MapState.version`, plus the console's `_State` counter when it exists). The
`/commit` reply carries `map_version` and `state_version` (null without a
`_State` node). The server logs which one it uses at startup.

Optional heracles features. The server logs at startup which are missing; none
is needed to query and commit:

| heracles function | turns on |
| --- | --- |
| `create_vector_indexes`, `insert_trajectory_frames_to_db`, `insert_frame_edges` | `python -m openset_query.ingest_keyframes` and `TrajectoryFrameIndex.ingest` (TrajectoryFrame nodes) |
| `query_similar_nodes` | `TrajectoryFrameIndex` vector search |
| `insert_frame_edges` | `DEPICTS` (TrajectoryFrame -> Object) provenance edges after a commit |
| `create_epistemic_indexes` | indexes for the evidence store (`EvidenceStore.ensure_schema`) |
| `constants.ADMISSION_BELIEF` | only a drift-guard test |

Without them those tools stop with an error naming the missing function, and the
server's ranking uses the SigLIP2 embeddings it keeps on the keyframe nodes.

## Origin and changes

Extracted from the `sam3_frontend` of the (private) agentic_navigation project,
at the version deployed on the ADT4 base station. Compared with that version:

- the package is `openset_query` (was `agentic_navigation.sam3_frontend` plus
  the `perception`, `evidence` and `eval` pieces it imports);
- images are read and written through Hydra's `hydra_python.image_store` instead
  of the fork-only `hydra_image_store` / `hydra_image_store_adt4` packages; the
  SAM 3 write-back helpers live in `openset_query.store_writeback`. Behaviour is
  unchanged (the old package re-exported the same code);
- `TrajectoryFrameIndex.lookup_near*` pass `LIMIT` as a parameter (Neo4j 5.2x
  rejected the old query);
- it runs on heracles master instead of the ADT4 heracles fork (see above);
  `/commit` returns `state_version: null` when the database has no console
  `_State` node;
- `HERACLES_NEO4J_PASSWORD` is required (there is no default password), and the
  URI defaults to `bolt://localhost:7687`;
- the HTTP contract, the admission policies and their flags are unchanged.

## Tests

```bash
pip install pytest
PYTHONPATH=<hydra>/hydra/python/src python -m pytest -q
```

Tests that load model weights run only with `OPENSET_QUERY_MODEL_TESTS=1`;
Neo4j tests run only with `NEO4J_URI` (and `NEO4J_AUTH=user/password`) pointing
at a throwaway database, which they write to; tests marked `adt4` need `heracles`
and `spark_dsg`.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). SAM 3 and the model
weights are third-party and under their own licenses.
