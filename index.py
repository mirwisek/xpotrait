from fastapi import FastAPI, HTTPException, UploadFile, Form
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import uuid
from pathlib import Path
from core.api_run import generate_ai_video


app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up static files to serve HTML and related assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Output directory for generated files
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class VideoRequest():
    source_image: UploadFile
    driving_video: UploadFile


@app.get("/")
def read_root():
    """Serve the main HTML file."""
    print('Root served')
    return FileResponse("static/index.html")

@app.post("/generate-video/")
async def generate_video(
    source_image: UploadFile, 
    driving_video: UploadFile
):
    """
    Generate a video using the provided source image and driving video.
    The script runs with static parameters except for the input files.
    """
    # Define static script parameters
    model_config = "config/cldm_v15_appearance_pose_local_mm.yaml"
    resume_dir = "checkpoint/model_state-415001.th"
    seed = 999
    uc_scale = 5
    best_frame = 36
    out_frames = -1
    num_mix = 4
    ddim_steps = 30

    # Save uploaded files
    source_image_path = Path(OUTPUT_DIR) / f"source_image_{uuid.uuid4().hex}.png"
    driving_video_path = Path(OUTPUT_DIR) / f"driving_video_{uuid.uuid4().hex}.mp4"

    print(f"Source image path: {source_image.filename}")
    print(f"Driving video path: {driving_video.filename}")

    with open(source_image_path, "wb") as img_file:
        img_file.write(await source_image.read())
    with open(driving_video_path, "wb") as video_file:
        video_file.write(await driving_video.read())

    # Command to run the script
    command = [
        "python3", "core/test_xportrait.py",
        "--model_config", model_config,
        "--output_dir", str(OUTPUT_DIR),
        "--resume_dir", resume_dir,
        "--seed", str(seed),
        "--uc_scale", str(uc_scale),
        "--source_image", str(source_image_path),
        "--driving_video", str(driving_video_path),
        "--best_frame", str(best_frame),
        "--out_frames", str(out_frames),
        "--num_mix", str(num_mix),
        "--ddim_steps", str(ddim_steps),
    ]


    try:
        # Run the script as a subprocess
        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        # print(result.stdout)
        output_video_path = generate_ai_video(model_config=model_config, output_dir=str(OUTPUT_DIR), resume_dir=resume_dir, seed=seed, 
                          uc_scale=uc_scale, source_image=str(source_image_path), driving_video=str(driving_video_path), 
                          best_frame=best_frame, out_frames=out_frames, num_mix=num_mix, ddim_steps=ddim_steps)

        # Ensure the output video exists
        # generated_video_files = list(Path(OUTPUT_DIR).glob("*.mp4"))
        # if not generated_video_files:
        #     raise HTTPException(status_code=500, detail="Failed to generate video.")
        if output_video_path is None:
            raise HTTPException(status_code=500, detail="Failed to generate video.")
        else:
            print('Generated Video')
            return {"video_url": f"{output_video_path}"}

        # print('Generated Video')
        # generated_video_url = str(generated_video_files[-1])  # Get the most recent video
        # return {"video_url": f"/{generated_video_url}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")
    


@app.get("/{file_path:path}")
async def serve_file(file_path: str):
    """Serve files from the outputs directory."""
    full_path = Path(OUTPUT_DIR) / file_path
    if full_path.exists():
        return FileResponse(full_path)
    raise HTTPException(status_code=404, detail="File not found")
