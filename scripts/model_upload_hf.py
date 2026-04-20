from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model/predictive_maintenance_model.pkl",
    path_in_repo="predictive_maintenance_model.pkl",
    repo_id="akhurana-hf/Predictive-Maintenance-Model",
    repo_type="model"
)
