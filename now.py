import subprocess

def extract_chunk(input_file, output_file="chunk3.mp4", start_minute=45, target_size_mb=150):
    # Get overall bitrate using ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file
    ]
    bitrate_str = subprocess.check_output(probe_cmd).decode().strip()
    bitrate = int(bitrate_str)  # bits per second

    # Calculate duration that fits into target size
    target_size_bytes = target_size_mb * 1024 * 1024
    duration_seconds = (target_size_bytes * 8) / bitrate

    # Run ffmpeg to extract chunk
    ffmpeg_cmd = [
        "ffmpeg",
        "-ss", str(start_minute * 60),   # start at 45 minutes
        "-i", input_file,
        "-c", "copy",
        "-t", str(int(duration_seconds)),
        output_file
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Created {output_file} (~{target_size_mb} MB) starting at {start_minute} minutes.")

# Example usage
extract_chunk("C:\\Users\\shivu\\Desktop\\projects\\cricket_highlights\\v1.mp4", output_file="chunk3.mp4", start_minute=45, target_size_mb=150)
