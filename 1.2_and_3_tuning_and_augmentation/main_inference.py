from src.inference import run_inference
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    run_inference(
        model_path=os.path.join(base_dir, "models", "pipeline_models", "best_lr0.001_wd0.0001_adam_ep100.pth"),
        test_zip_path="data/X_test.zip",
        output_path=os.path.join(base_dir, "data", "base.csv"),
        batch_size=32
    )