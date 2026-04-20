from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model/model.joblib",
    path_in_repo="model.joblib",
    repo_id="akhurana-hf/Predictive-Maintenance-Model",
    repo_type="model"
)
