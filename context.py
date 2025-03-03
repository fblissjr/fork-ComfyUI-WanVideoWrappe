import numpy as np
from typing import Callable, Optional, List


# Improved FPS sampling function based on Apollo paper recommendations
def sample_video_fps(total_frames, original_fps, target_fps, max_frames=None):
    """
    Sample frames at a consistent frames per second rate

    Args:
        total_frames: Total number of frames in the video
        original_fps: Original frames per second of the video
        target_fps: Target frames per second to sample at
        max_frames: Maximum number of frames to sample (optional)

    Returns:
        List of frame indices to sample
    """
    # Calculate video duration in seconds
    video_length_sec = total_frames / original_fps

    # Calculate how many frames to sample based on target FPS
    num_frames_to_sample = int(video_length_sec * target_fps)

    if max_frames and num_frames_to_sample > max_frames:
        # If exceeding max frames, we cap at max_frames
        num_frames_to_sample = max_frames

    # Calculate sampling interval in terms of original frames
    if num_frames_to_sample >= total_frames:
        # If we need all frames or more, just return all frames
        return list(range(total_frames))

    sample_interval = total_frames / num_frames_to_sample
    frame_indices = [
        min(int(i * sample_interval), total_frames - 1)
        for i in range(num_frames_to_sample)
    ]

    return frame_indices


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1


def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        # 1) subtract each element by start_val to move vals relative to the start of all frames
        # 2) add num_frames and take modulus to get adjusted vals
        window[i] = ((window[i] - start_val) + num_frames) % num_frames


def shift_window_to_end(window: list[int], num_frames: int):
    # 1) shift window to start
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        # 2) add end_delta to each val to slide windows to end
        window[i] = window[i] + end_delta


def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes


def uniform_looped(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


# from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
def uniform_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            windows.append(
                [
                    e % num_frames
                    for e in range(j, j + context_size * context_step, context_step)
                ]
            )

    # now that windows are created, shift any windows that loop, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window is rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][
                roll_idx
            ]  # roll_val might not be 0 for windows of higher strides
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window (cyclical) is missing roll_val
            if roll_val not in windows[(win_i + 1) % len(windows)]:
                # need to insert new window here - just insert window starting at roll_val
                windows.insert(
                    win_i + 1, list(range(roll_val, roll_val + context_size))
                )
        # delete window if it's not unique
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # reverse delete_idxs so that they will be deleted in an order that doesn't break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)
    return windows


def static_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = context_size - context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + context_size
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + context_size)))
            break
        windows.append(list(range(start_idx, start_idx + context_size)))
    return windows


# FPS-based uniform standard scheduler
def uniform_standard_fps(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    target_fps: float = 2.0,
    original_fps: float = 30.0,
    closed_loop: bool = True,
):
    """
    Creates context windows based on consistent FPS sampling as recommended in the Apollo paper

    Args:
        step: Current step in the diffusion process
        num_steps: Total number of steps in the diffusion process
        num_frames: Total number of frames to process
        context_size: Size of each context window in frames
        context_stride: Stride between context windows
        context_overlap: Overlap between context windows
        target_fps: Target frames per second to sample at
        original_fps: Original frames per second of the video
        closed_loop: Whether to allow looping at the end
    """
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    # Calculate time represented by each frame
    # In latent space, note that we're working with latent frames
    latent_stride = 4  # 4 pixel frames = 1 latent frame in WanVideo
    seconds_per_latent_frame = latent_stride / original_fps
    video_length_sec = num_frames * seconds_per_latent_frame

    # Calculate window timing in seconds
    window_duration_sec = context_size * seconds_per_latent_frame
    window_stride_sec = (context_size - context_overlap) * seconds_per_latent_frame

    # Generate windows based on temporal position
    for window_start_sec in np.arange(0, video_length_sec, window_stride_sec):
        # Calculate frame indices for this time window
        start_frame = int(window_start_sec / seconds_per_latent_frame)
        end_frame = min(start_frame + context_size, num_frames)

        # Ensure we maintain context_size frames per window when possible
        if end_frame - start_frame < context_size and start_frame > 0:
            start_frame = max(0, end_frame - context_size)

        # Create the window of sequential frames
        window_frames = list(range(start_frame, end_frame))

        # If we need to sample at a specific FPS within this window
        if (
            target_fps * seconds_per_latent_frame < 1.0
            and end_frame - start_frame > context_size
        ):
            # Calculate frames at target FPS
            window_duration = (end_frame - start_frame) * seconds_per_latent_frame
            frames_at_target_fps = int(window_duration * target_fps)
            frames_at_target_fps = min(frames_at_target_fps, context_size)

            if frames_at_target_fps < end_frame - start_frame:
                # Sample frames evenly within the window
                step_size = (end_frame - start_frame) / frames_at_target_fps
                window_frames = [
                    start_frame + int(i * step_size)
                    for i in range(frames_at_target_fps)
                ]
                # Make sure we include the end frame for continuity
                if window_frames[-1] != end_frame - 1:
                    window_frames[-1] = end_frame - 1

        windows.append(window_frames)

        if end_frame >= num_frames:
            break

    # Handle window overlap and consistency checks
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window is missing roll_val
            if win_i + 1 < len(windows) and roll_val not in windows[win_i + 1]:
                # Add a new window to cover the gap
                windows.insert(
                    win_i + 1,
                    list(range(roll_val, min(roll_val + context_size, num_frames))),
                )

        # Delete duplicate windows
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # Remove duplicates
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)

    return windows


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform_looped":
        return uniform_looped
    elif name == "uniform_standard":
        return uniform_standard
    elif name == "static_standard":
        return static_standard
    elif name == "uniform_standard_fps":
        return uniform_standard_fps
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )
