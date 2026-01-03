import detect

# Run the detector with higher stability and longer speak cooldown
if __name__ == '__main__':
    # camera_index, model_path, scaler_path, stability_frames, tts_rate, tts_volume, speak_cooldown
    detect.main(camera_index=1, model_path='asl_sign_model.pkl', scaler_path='asl_scaler.pkl', stability_frames=5, tts_rate=150, tts_volume=0.9, speak_cooldown=4.0)
