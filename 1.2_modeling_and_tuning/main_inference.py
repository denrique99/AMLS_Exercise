from src.inference import run_inference

if __name__ == "__main__":
    run_inference(
        model_path="models/pipeline_models/best_lr0.001_wd0.0001_adam_ep100_after_meeting.pth",
        test_zip_path="data/X_test.zip",
        output_path="data/base.csv",
        batch_size=32
    )