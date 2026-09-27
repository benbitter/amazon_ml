# Amazon ML Challenge 2026 — Business Entity Resolution

```
amazon-ml-er/
├── code/business_entity_resolution/   # the pipeline (goes into the submission zip)
│   ├── src/                           # normalize → blocking → features → model → decision
│   ├── tests/                         # synthetic smoke test
│   ├── README.md                      # how to reproduce end to end
│   └── requirements.txt
├── notebooks/kaggle_runner.ipynb      # run everything on Kaggle
├── output/                            # matching_results.tsv + candidate_pairs.tsv land here
├── Documentation_draft.md             # methodology write-up → copy into Documentation_template.md
├── make_zip.py                        # builds <team>_submission.zip in the required layout
└── dataset/                           # put the provided train/ and test/ here (not committed)
```

## Quick start

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python -m src.run_pipeline --data-dir ../../dataset --out-dir ../../output --work-dir artifacts
cd ../..
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Upload `output/matching_results.tsv` to the portal.

At the end, fill in `Documentation_template.md` (start from `Documentation_draft.md`), then build the submission zip:

```bash
python make_zip.py --team <team_name>
```
