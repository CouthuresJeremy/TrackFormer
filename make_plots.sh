
script_path=/home/couthures/Bureau/TrackParameters/TrackFormer/scripts/run_analysis.py

version=487

# /home/couthures/anaconda3/envs/ML/bin/python ${script_path} --model-dir "Models/TrackML/version_${version}" --output-dir command_results


# version=499
# /home/couthures/anaconda3/envs/ML/bin/python ${script_path} --model-dir "Models/TrackML/version_${version}" --output-dir "Results/TrackML/rdphiz_n_hits_3_qopT_pz_POC_TrackML_zenodo_full_trained_5_hits" --dataset-name TrackML_zenodo_full
# /home/couthures/anaconda3/envs/ML/bin/python ${script_path} --model-dir "Models/TrackML/version_${version}" --min-hits 5 --dataset-name TrackML_zenodo_full

# version=464
# /home/couthures/anaconda3/envs/ML/bin/python ${script_path} --model-dir "Models/TrackML/version_${version}" --output-dir "Results/TrackML/rdphiz_n_hits_3_qopT_pz_POC_TrackML_zenodo_full_trained_7_hits" --dataset-name TrackML_zenodo_full
# /home/couthures/anaconda3/envs/ML/bin/python ${script_path} --model-dir "Models/TrackML/version_${version}" --min-hits 7 --dataset-name TrackML_zenodo_full

version=534
/home/couthures/anaconda3/envs/ML/bin/python ${script_path} --model-dir "Models/TrackML/version_${version}" --output-dir "Results/TrackML/rdphiz_n_hits_3_qopT_pz_POC_TrackML_zenodo_eta_3" --dataset-name TrackML_zenodo --max-abs-eta 3


python scripts/run_analysis.py --model-dir "Models/TrackML/lightning_logs/version_499" --output-dir "Results/TrackML/rdphiz_n_hits_7_qopT_pz_POC_TrackML_zenodo_full_trained_5_hits" --dataset-name TrackML_zenodo_full  --min-hits 7

python scripts/run_analysis.py --model-dir "Models/TrackML/lightning_logs/version_392" --output-dir "Results/TrackML/rdphiz_n_hits_3_qopT_pz_POC_TrackML_zenodo_full_eta_1_32_16" --dataset-name TrackML_zenodo_full  --max-abs-eta 1 --publication
