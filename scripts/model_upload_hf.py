from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model/predive-maintenance_model.pkl",
    path_in_repo="predictive-maintenance_model.pkl",
    repo_id="akhurana-hf/predictive_maintenance/engine-failure-model",
    repo_type="model"
)
