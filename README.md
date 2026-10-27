# Open-set synthetic anomalies for cashew (VisA)

Code to train the anomaly generator (SD 1.5 inpainting + IP-Adapter Plus), to build the synthetic
training sets of the open-set (leave-one-type-out) experiment, to train and score UniNet on them, and to make the
tables; plus sheets of the configuration-1 synthetic training items of all 10 seeds and pages of all real reference
anomalies.

## Layout

| Path | Contents |
|---|---|
| `src/` | generator model code (IP-Adapter, SD pipelines), dataset, mask / crop / placement utilities |
| `scripts/train_anomagic.py` | generator training |
| `scripts/open_set_launch.py` | one launcher for the open-set experiment: plan, generate, refine, dtd, cutmix, check, train, tables |
| `scripts/open_set_steps.py` | the individual steps (see its docstring); `pregenerate_synthetic.py` holds placement, generation settings and reference / donor pools; `data_sources.py` holds the dataset locations, the generator checkpoint path and the donor-listing / mask helpers |
| `scripts/train_uninet.py` | UniNet launcher with the fixed configurations (`--hero caao`, `m1`, `mode3`) |
| `scripts/uninet/run_uninet_cashew.py` | the UniNet training / evaluation runner |
| `scripts/score_runner.py`, `open_set_scores.py` | runner variant that also writes per-image scores |
| `scripts/open_set_tables.py`, `open_set_typeincl.py`, `open_set_closed*.py`, `open_set_mix.py` | tables and the supplementary experiments (type-inclusive, closed-set control, configuration 6: the four synthetic methods mixed) |
| `data/` | fixed inputs the pipeline reads: `splits.json` (train / test normals and the spare normals the canvases are drawn from), `real_anomalies.json` (the 84 single-label real cashew anomalies: id, VisA label, type, image path), `anomaly_plan_b3.json` (the 3 real anomalies per type of each seed) |
| `configs/` | records of the settings used: `generator_settings.json` (generation: checkpoint, mask factor, noise strength, steps, guidance, CFG mode), `generator_training_config.json` (generator training run) |
| `references/` | pages showing every real reference anomaly used in any configuration (1-5, all 10 seeds), cropped around the defect with the mask outlined in cyan. One page per anomaly type in `target_object/<n>_<type>.png` = the VisA cashew defects (target-object references, 84 in total), and in `cross_object/<n>_<type>.png` = the donor defects (cross-object references, 496 in total: 298 Real-IAD fire_hood; VisA 78 pipe_fryum, 54 macaroni1, 51 macaroni2, 15 fryum; longer lists span several pages, e.g. `1_holes_1of7.png` … `1_holes_7of7.png`). Configurations 1-4 use the 236 matched donors, configuration 5 the unrestricted pool of 492. Each donor is tagged `c1-4` (matched donor of configurations 1-4), `c5` (only in the unrestricted pool of configuration 5) or `c1-5` (both) |
| `configuration1_sheets/seed_<s>/` | per seed: one sheet per anomaly type with all 9 training items (target reference, cross reference, clean canvas, target-object, cross-object, CutMix, DTD; the 3 placements of each reference in consecutive rows). Diffusion and DTD images are the training images. CutMix is shown once per item with its patch taken from another canvas of the same type's 9 canvases; in training the CutMix patch was drawn per fold from that fold's 45 canvases (same code, bit-exactly reproduced) |

Naming: configuration *k* is called stage *k* in the code; arms carry the suffix `_s<k>` (e.g. `diffusion_in_s1`).
Configuration 1 uses image set `s2`.

## Not included

- Datasets and derived inputs, at these paths relative to the repository root
  (`C` = `anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew`):
  - VisA cashew: `C/Data`, `C/image_anno.csv`, BiRefNet foreground masks `C/birefnet_masks`, `C/birefnet_masks_anomaly`, `C/experiment_UniNet/source`, `C/experiment_UniNet/synthetic`,
    `C/experiment_ResNet/source`
  - VisA donor objects: `anomverse_extension/datasets/VisA_validation_dataset/restVisA/VisA_20220922/{fryum,macaroni1,macaroni2,pipe_fryum}`
  - Real-IAD fire_hood: `anomverse_extension/datasets/full_training_dataset/realiad_1024/fire_hood`
  - generator training set: `anomverse_extension/datasets/full_training_dataset/{master_training.json,captions_from_master.json}` and its images
  - DTD: `datasets/dtd`
- Pretrained models from Hugging Face: `runwayml/stable-diffusion-inpainting`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`,
  `h94/IP-Adapter`, `timm/convnext_base.dinov3_lvd1689m`; torchvision `wide_resnet50_2`.
- The trained generator checkpoint (`results/generator_nocashew_20k_constlr/checkpoint_20000`); its training settings
  are in `configs/generator_training_config.json`.

## Running

```bash
pip install -r requirements.txt

# generator (settings: configs/generator_training_config.json; --no-eval skips the old evaluation pipeline)
python scripts/train_anomagic.py --no-live-viewer --no-eval ...

# open-set experiment: smoke test, then the full run
python scripts/open_set_launch.py --profile server --smoke
python scripts/open_set_launch.py --profile server --seeds 42 123 7 99 256 11 22 33 44 55
```

## UniNet runner

`scripts/uninet/run_uninet_cashew.py` is the version used for the reported runs with two edits: the lines that locate the repository root and `data/splits.json`, and open-set runs (`--test-anomaly-list` with `--synthetic-epochs` or `--baseline-extra-list`) no longer load the earlier fixed synthetic set, which they replaced and never used. Training and evaluation are unchanged (checked: a configuration-1 run gets identical training and test inputs with and without that set).

## Data sources

The sheets and `references/` contain VisA cashew images, VisA donor images and Real-IAD donor images. Their use is subject to the original
datasets' licences and terms (VisA: CC BY-NC-SA 4.0; Real-IAD: its terms of use).
