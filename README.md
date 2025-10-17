# Corner Maze Encoder Pretrain

This repository explores how to compress visual observations collected from a rodent-scale corner maze into a compact latent space that can condition downstream reinforcement learning (RL) agents. The raw observations are stereo renderings (left- and right-eye) of the 2s2c behavioral task inside a Fusion360 model of the maze. Each pose in an 11×11 spatial grid produces four views aligned with the cardinal directions, except at corners where diagonal headings (NE, NW, SE, SW) are captured. All renders are flattened into single-channel (B/W) images and paired so that a single tensor contains both eyes.

## Project Goals
- Build preprocessing utilities that organize the stereo renders, flatten them, and stack them into tensors suitable for PyTorch.
- Prototype multiple encoder architectures (CNNs and alternative tensor representations) to learn latent features that uniquely identify maze poses.
- Evaluate the learned encoders by decoding pose identity and by feeding the latent vectors into lightweight RL agents in a MiniGrid-style simulator of the 2s2c task.

## Tentative Planned Repository Structure
```
corner-maze-encoder-pretrain/
├── data/                # Placeholder tree kept empty in git
│   └── images/          # Drop local stereo renders here (do not commit)
├── notebooks/           # Exploratory data analysis and prototyping
├── src/
│   ├── datasets/        # Dataset loaders and preprocessing pipelines
│   ├── models/          # Encoder architectures and training utilities
│   └── rl/              # Interfaces to downstream RL experiments
├── tests/               # Unit and regression tests
└── README.md
```
The `data/` directory lives in the repo with placeholder `.gitkeep` files so collaborators share the same layout while keeping bulky renders local.

## Getting Started
1. **Activate the shared environment**
   ```bash
   source ~/venvs/ai-venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   VS Code is already configured (via `.vscode/settings.json`) to use this interpreter automatically.
2. **Ingest image renders**  
   Place raw left/right camera renders under `data/images/<pose_id>/<view>.png`. The `data/` tree is committed only with empty placeholder files; keep your local renders uncommitted (e.g., via `git status` checks or temporary stash) so the repository history stays lightweight. A preprocessing script (to be added) will flatten the images, pair both eyes, and emit tensors into `data/processed/`.
3. **Run encoder pretraining**  
   Training entry points will live under `src/` and will expose CLI interfaces for experimenting with different tensor layouts (e.g., concatenated eye channels vs. spatial stacking).

## Current Utilities
- `src/image_stack.py` loads stereo PNG pairs, converts them to blurred grayscale tensors, mirrors the right eye, and stacks them into `(pairs, 2, H, W)` PyTorch tensors. It also exposes `visualize_stereo_pair` for quick overlays (left eye tinted blue, right eye red).
- `notebooks/image_stack.ipynb` offers a notebook version of the same pipeline for interactive exploration (you may need to reload the notebook after code updates).

## Experimental Roadmap
- Compare latent spaces produced by CNN encoders with variants that exploit stereo correspondences (e.g., Siamese branches, shared weights, or 3D convolutions).
- Benchmark contrastive objectives (InfoNCE, triplet loss) against classification-style supervision for pose identification.
- Integrate the resulting latent encoders into MiniGrid RL loops to study how representation fidelity impacts learning on the 2s2c task.

## Contributing
Early contributions should focus on:
- Creating reproducible data preprocessing scripts.
- Implementing baseline encoders and training loops with PyTorch Lightning or vanilla PyTorch.
- Adding tests that check tensor shapes, pose coverage, and encoder invariances.

Please open an issue or draft pull request to coordinate efforts.
