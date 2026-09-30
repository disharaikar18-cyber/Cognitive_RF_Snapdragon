import qai_hub as hub

def compile_and_profile():
    print("Connecting to Qualcomm AI Hub...")
    device = hub.Device("Snapdragon X Elite CRD")
    
    print("Submitting compilation job to AI Hub...")
    compile_job = hub.submit_compile_job(
        model="rf_classifier.onnx",
        device=device,
        options="--target_runtime onnx",
        input_specs=dict(spectrogram=(1, 1, 64, 64))
    )
    
    target_model = compile_job.get_target_model()
    
    print("Submitting profiling job to measure NPU performance...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device
    )
    
    compiled_model = compile_job.download_target_model("rf_classifier_snapdragon.onnx")
    print("Successfully compiled and downloaded: rf_classifier_snapdragon.onnx")
    
    profile_data = profile_job.download_profile()
    print("\n--- QUALCOMM NPU PROFILING RESULTS ---")
    print(profile_data)

if __name__ == "__main__":
    compile_and_profile()