import sys

sys.path.insert(0, '/workspace/core/')

from test_xportrait import get_inference

def generate_ai_video(
    # Model parameters
    model_config="model_lib/ControlNet/models/cldm_v15_video_appearance.yaml",
    reinit_hint_block=False,
    sd_locked=True,
    only_mid_control=False,
    control_type="appearance_pose_local_mm",
    control_mode="controlnet_important",
    wonoise=True,
    
    # Training parameters
    local_rank=0,
    world_size=1,
    seed=42,
    use_fp16=True,
    compile=False,
    eta=0.0,
    ema_rate=0,
    
    # Inference parameters
    initial_facevid2vid_results=None,
    ddim_steps=1,
    uc_scale=5,
    num_drivings=16,
    output_dir=None,
    resume_dir=None,
    source_image="",
    more_source_image_pattern="",
    driving_video="",
    best_frame=0,
    start_idx=0,
    skip=1,
    num_mix=4,
    out_frames=0
):
    # Create args dictionary with all parameters
    args = {
        'model_config': model_config,
        'reinit_hint_block': reinit_hint_block,
        'sd_locked': sd_locked,
        'only_mid_control': only_mid_control,
        'control_type': control_type,
        'control_mode': control_mode,
        'wonoise': wonoise,
        'local_rank': local_rank,
        'world_size': world_size,
        'seed': seed,
        'use_fp16': use_fp16,
        'compile': compile,
        'eta': eta,
        'ema_rate': ema_rate,
        'initial_facevid2vid_results': initial_facevid2vid_results,
        'ddim_steps': ddim_steps,
        'uc_scale': uc_scale,
        'num_drivings': num_drivings,
        'output_dir': output_dir,
        'resume_dir': resume_dir,
        'source_image': source_image,
        'more_source_image_pattern': more_source_image_pattern,
        'driving_video': driving_video,
        'best_frame': best_frame,
        'start_idx': start_idx,
        'skip': skip,
        'num_mix': num_mix,
        'out_frames': out_frames
    }
    
    if output_dir is None:
        raise ValueError("output_dir is required")

    return get_inference(args) # Returns output path of the video
