from src.inference import run_inference

if __name__ == "__main__":
    run_inference(
        model_path="models/tuning_approach_2/best_lr0.001_weightdecay0.0001.pth",
        test_zip_path="data/X_test.zip",
        output_path="data/base.csv",
        batch_size=32
    )