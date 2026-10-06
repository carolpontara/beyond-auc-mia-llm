# Federated IDS membership inference audit

Executed instantiation of the multidimensional audit for a federated intrusion detector:
Flower simulation with ten clients on one machine, PyTorch MLP, CIC-IDS2017 and CICIoT2023,
Dirichlet label skew at alpha = 0.1, 0.5 and 1.0, aggregators FedAvg, FedProx, SCAFFOLD and
FedAdam, five membership scores audited at rounds 10, 50 and 100.

## Design

| Element | Choice |
|---|---|
| Data | Deduplicated pool per dataset, Benign capped at 20,000 records and each attack class at 6,000, split once, stratified, into member (40%), non-member (20%), reference (20%) and test (20%) sets. Features: signed log1p, standardised with reference statistics, clipped to [-10, 10]. CIC-IDS2017 labels grouped into 7 classes (Infiltration and Heartbleed dropped, under 40 records). |
| Federation | 10 clients, full participation, 100 rounds, 1 local epoch, SGD lr 0.05, batch 32, MLP 128-64. |
| Partition | Dirichlet label skew of the member set, partition seed 42, identical for every aggregator and every training seed (`partitions/*.npz`). |
| Aggregators | FedAvg; FedProx (mu = 0.01); SCAFFOLD option II (custom strategy on Flower FedAvg, client control variates kept in the Flower node state); FedAdam (server lr 0.01, tau 1e-3). Only the aggregator changes between cells. |
| Seeds | 0, 1, 2 (model initialisation and client batch order). |
| Stored artefacts | Global model every 10 rounds; broadcast model and every client's local model at rounds 10, 50 and 100 (`updates_r*.npz`). |
| Attacks | `loss`, `refgap` (reference model trained on the reference split), `gradnorm`, `gradalign` (server-side cosine between the per-sample descent direction and each client update), `trajectory` (mean standardised score over rounds 10 to 100). See `attacks.py`. |
| Calibration | Thresholds from non-member scores only; a record is exposed at rate a if its score is at least the threshold giving FPR <= a. |
| Outputs | AUC, TPR at 1% and 5% FPR, identities of exposed records, fraction of the training data exposed, union of attacks (descriptive and Bonferroni-recalibrated to a 1% global FPR), Jaccard stability across seeds against its chance level, per-class TPR with FPR, per-client exposure against the class-matched FPR. |
| Positive control | 200 member and 200 non-member records with flipped labels (`make_canary.py`), CIC-IDS2017, alpha = 0.5, seed 0, the four aggregators, same pipeline. |

## Reproduce

```bash
pip install -r requirements.txt
python prepare_data.py --dataset cicids2017 --raw <dir with the 8 CIC-IDS2017 parquet files> --out ~/fedids_work/data/cicids2017.npz
python prepare_data.py --dataset ciciot2023 --raw <dir with train.parquet> --out ~/fedids_work/data/ciciot2023.npz
./run_grid.sh                       # trains and scores the 72 runs
python make_canary.py && DATASETS=cicids2017_canary SEEDS=0 ALPHAS=0.5 ./run_grid.sh
python analyze.py --out results && python canary_report.py && python report.py --res results
```

Parquet sources: CIC-IDS2017 from the Hugging Face dataset `Ariasyah/cic-ids-2017`;
CICIoT2023 from `casscloud/CIC-IoT-2023` (random split, balanced subsample).

`results/` holds every number reported in the paper: `runs.csv` (one row per run, attack and
round), `union.csv`, `stability.csv`, `per_class.csv`, `per_client.csv`,
`identified_samples.csv.gz` (pool index of every exposed record at 5% FPR, flag for 1% FPR),
`canary.csv`, LaTeX tables and figures. Checkpoints and client updates are not versioned
(about 250 MB); `run_grid.sh` regenerates them.

## Measured results (round 100, three seeds per cell, 72 runs)

- Utility: mean macro F1 per cell 0.948 to 0.979 on CIC-IDS2017, 0.619 to 0.688 on CICIoT2023.
- Discrimination: AUC 0.494 to 0.506 (CIC-IDS2017) and 0.498 to 0.513 (CICIoT2023) over all
  attacks, aggregators and concentrations; TPR at 1% FPR at most 1.24% and 1.43%.
- Aggregator: on CICIoT2023 FedAdam has a higher loss AUC than FedAvg in all nine paired
  alpha/seed cells (mean +0.0019); on CIC-IDS2017 the difference averages -0.0001 and changes sign.
- Stability: Jaccard of members flagged across seeds is far above chance, but flagged
  non-members are equally stable (within 0.014), so stability alone is not evidence of memorisation.
- Per class: high flagged rates (DoS, Mirai, DDoS) are matched by the same-class non-member FPR.
- Positive control: canary members are ranked above canary non-members (loss AUC 0.678 to 0.741),
  but thresholds calibrated on the regular non-member population flag none of them, and the
  reference gap flags almost all canaries, members and non-members alike. The control confirms
  ranking sensitivity, not threshold-level detection.

## Limitations

Class-capped subsets of both datasets; CICIoT2023 split is random, not temporal; small MLP;
three seeds over one partition per alpha; passive attacks without shadow models; no DP.

