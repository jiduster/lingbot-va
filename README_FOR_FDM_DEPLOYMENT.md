# FDM Deployment Notes for LingBot-VA on Pika

This note is intended for the deployment-side machine and for the agent that
will continue the real-robot integration work.

## Current Status

We have trained a Pika single-task LingBot-VA checkpoint with additional FDM
post-training. The open-loop visualization on the training data looked usable.

The current policy server can load and run this FDM-trained transformer
checkpoint, but it does not yet implement the full asynchronous FDM deployment
pipeline described in the paper's Section 3.4. In the current server:

- `VA_Server._infer()` runs a video latent branch with `action_mode=False`.
- `VA_Server._infer()` then runs an action branch with `action_mode=True`.
- The websocket policy API returns only `dict(action=action)`.
- The predicted video latent is not exposed as a deployment state, and it is
  not used by a dedicated FDM feedback/state-update protocol.

So the FDM checkpoint can be deployed immediately for action prediction, but
the full paper-style FDM asynchronous inference pipeline still requires client
and server changes.

## Important Paths

Main repository:

```bash
/data/home/zyh/lingbot-va
```

Base LingBot-VA / Wan2.2 assets:

```bash
/mnt/ceph/ckpt/lingbot-va-posttrain-robotwin
```

Current FDM post-training checkpoint example:

```bash
/data/home/zyh/lingbot-va/train_out/pika_fdm/checkpoints/checkpoint_step_1000
```

The FDM checkpoint directory contains the trained `transformer/`. The base
model directory must still provide the VAE, tokenizer, and text encoder.

## Environment

Use this repository's own virtual environment, not a global conda or another
project's environment:

```bash
cd /data/home/zyh/lingbot-va
./.venv/bin/python -V
```

The shell's default `python` may point to a different environment. Prefer
`./.venv/bin/python` or ensure the local uv environment is active.

## Loading the FDM Checkpoint for Deployment

For deployment, keep:

```python
va_pika_cfg.wan22_pretrained_model_name_or_path = "/mnt/ceph/ckpt/lingbot-va-posttrain-robotwin"
```

and add the transformer checkpoint override:

```python
va_pika_cfg.transformer_checkpoint_path = (
    "/data/home/zyh/lingbot-va/train_out/pika_fdm/checkpoints/checkpoint_step_1000"
)
```

This keeps VAE/tokenizer/text encoder loaded from the base model root while
loading the fine-tuned FDM transformer from the checkpoint directory.

The current `wan_va_server.py` supports `transformer_checkpoint_path` in the
config object, but the CLI does not yet expose it as a command-line override.
The simplest deployment path is to set it in `wan_va/configs/va_pika_cfg.py`.

## Basic Server Launch

For Pika, use:

```bash
cd /data/home/zyh/lingbot-va

NGPU=1 CONFIG_NAME=pika bash script/run_launch_va_server_sync.sh
```

For multi-GPU/FSDP serving:

```bash
cd /data/home/zyh/lingbot-va

NGPU=4 CONFIG_NAME=pika MASTER_PORT=29517 bash script/run_launch_va_server_sync.sh
```

Check that the robot client sends the expected observation keys matching:

```python
[
    "observation.images.scene",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
```

The action output is still the LingBot-VA 30-D action vector after postprocess.
For Pika, the useful action channels are:

```python
left arm joints:  action[14:20]
right arm joints: action[21:27]
grippers:         action[28], action[29]
```

The Pika training data used 12 arm joints plus 2 gripper values. The config maps
those values into LingBot-VA's 30-D action layout.

## What Is Missing for Full Paper Section 3.4

The full FDM asynchronous deployment pipeline requires more than loading an FDM
checkpoint. It needs the execution thread and the inference thread to overlap,
and it needs feedback from real observations to update the model state.

The current implementation does not yet provide:

- a client-side action buffer/executor thread,
- a client-side observation queue collected while actions are being executed,
- a websocket request type that sends recent real observations plus executed
  actions to the server,
- a server method that uses FDM to update the internal latent/KV cache from
  real feedback,
- a server response that cleanly returns the next action chunk while execution
  continues,
- fallback behavior when inference is late or the action buffer runs dry.

## Recommended Implementation Plan

### Phase 1: Naive Async Client

First implement a simpler asynchronous client without changing model semantics.
This should already reduce blocking latency.

Client-side structure:

- `ObservationThread`: continuously reads camera frames and robot state.
- `ExecutorThread`: executes the current action chunk at the control rate.
- `InferenceThread`: requests the next action chunk from the server before the
  current chunk finishes.
- `ActionBuffer`: stores future actions and tracks buffer depth.

Behavior:

1. Send `reset=True` with the task prompt at episode start.
2. Request an initial action chunk synchronously.
3. Start executing the first chunk.
4. While executing, request the next chunk asynchronously.
5. If the next chunk returns in time, append it to the action buffer.
6. If inference is late, either hold the last safe action, slow down execution,
   or stop the robot depending on the safety policy.

This phase does not fully use the paper's FDM feedback mechanism, but it is a
low-risk step toward real-time deployment.

### Phase 2: FDM-Grounded Async Server Protocol

Then add the actual FDM feedback path.

Add a new request mode, for example:

```python
{
    "mode": "fdm_async_step",
    "recent_obs": [...],
    "executed_actions": ...,
    "prompt": "...",
}
```

Server-side responsibilities:

1. Encode the recent real observations into video latents.
2. Normalize and pack executed actions using the Pika action config.
3. Use the FDM branch (`action_mode=False`) to update the latent/KV cache.
4. Use the action branch (`action_mode=True`) to predict the next action chunk.
5. Return `dict(action=next_action_chunk, timing=...)`.

This is the part that corresponds to the paper's FDM-based asynchronous
pipeline. It should be implemented carefully because it changes the server
state machine.

## Server Code Areas to Inspect

Main server:

```bash
wan_va/wan_va_server.py
```

Important methods:

- `VA_Server._reset()`: resets cache and encodes the prompt.
- `VA_Server._encode_obs()`: encodes camera observations into VAE latents.
- `VA_Server._infer()`: current chunk prediction path.
- `VA_Server._compute_kv_cache()`: currently accepts real obs and state/action
  feedback, and may be the closest starting point for an FDM feedback update.
- `VA_Server.infer()`: websocket-facing server entry point.

Websocket wrapper:

```bash
wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py
wan_va/utils/sever_utils.py
```

These files define how client observations become calls to `model.infer(obs)`.

## Client-Side Requirements

The client must maintain a strict real-time boundary between action execution
and model inference.

Recommended client state:

- `latest_obs`: newest camera/state packet.
- `obs_queue`: timestamped observations collected during execution.
- `action_buffer`: queue of future actions.
- `current_chunk_id`: chunk currently being executed.
- `pending_request`: future/task for the next server request.
- `last_safe_action`: fallback action if inference misses its deadline.

Recommended safety checks:

- Stop if action buffer is empty for more than one control cycle.
- Stop if server request exceeds the configured timeout.
- Stop if observation timestamps are stale.
- Clamp joint/gripper commands to robot-safe limits.
- Verify action dimensionality and channel mapping before sending to robot.

## Profiling

Timing logs can be enabled with:

```bash
LINGBOT_VA_PROFILE_TIMINGS=1 \
NGPU=1 CONFIG_NAME=pika bash script/run_launch_va_server_sync.sh
```

The server will report timing for:

- prompt/T5 encoding,
- observation VAE encoding,
- video latent diffusion loop,
- action diffusion loop,
- VAE video decoding if used.

For real-robot deployment, the key timing is the action inference latency, not
the video decoding latency. Do not decode video on the policy server during
deployment unless it is needed for debugging.

## Known Caveats

- `enable_offload=True` saves VRAM but can add CPU/GPU transfer latency.
- T5 prompt encoding happens during reset, not every inference step.
- The current websocket server returns only action data.
- The FDM checkpoint improves the model's learned dynamics, but the full
  asynchronous FDM algorithm still requires protocol and client changes.
- Keep `wan22_pretrained_model_name_or_path` pointed at the base model root,
  not the fine-tuned checkpoint directory.
- Keep `transformer_checkpoint_path` pointed at the fine-tuned checkpoint
  directory.

