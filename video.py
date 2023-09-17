import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def find_edited_parts(video_path, start_time_in_seconds, end_time_in_seconds, ssim_threshold=0.95):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(start_time_in_seconds * fps)
    end_frame = int(end_time_in_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = end_frame - start_frame
    edited_frames = 0

    _, first_frame = cap.read()
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    output_frames = []

    with tqdm(total=total_frames, desc="Processing Video", unit="frames") as pbar:
        for frame_number in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ssim_score = ssim(first_frame_gray, gray_frame)

            if ssim_score < ssim_threshold:
                edited_frames += 1
                marked_frame = frame.copy()
                # Mark edited regions with red color
                marked_frame[np.where((first_frame_gray - gray_frame) != 0)] = [0, 0, 255]
                output_frames.append(marked_frame)
            else:
                output_frames.append(frame)

            pbar.update(1)

    cap.release()

    return edited_frames, total_frames, output_frames

video_path = "/home/joe/Downloads/10000000_277614841627191_5313153616506230972_n.mp4"
start_time = 30 
end_time = 4 * 60 + 40 
edited_frames, total_frames, output_frames = find_edited_parts(video_path, start_time, end_time)
percentage_of_editing = (edited_frames / total_frames) * 100

print(f"The percentage of editing in the video: {percentage_of_editing:.2f}%")

# Save the marked video output
output_path = "/home/joe/Downloads/marked_video_output.mp4"
height, width, _ = output_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

for frame in output_frames:
    output_video.write(frame)

output_video.release()
