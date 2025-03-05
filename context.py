# context.py
import numpy as np
from typing import Callable, Optional, List
from .utils import log


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
    step,
    num_steps,
    num_frames,
    context_size,
    context_stride,
    context_overlap,
    target_fps=2.0,
    original_fps=30.0,
    closed_loop=True,
    enable_frame_blending=True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    # Calculate window stride with guaranteed overlap
    window_stride = max(1, context_size - context_overlap)

    # Create windows with sequential frames for better interpolation
    for start_idx in range(0, num_frames, window_stride):
        end_idx = min(start_idx + context_size, num_frames)

        # For final window, adjust to maintain constant size if possible
        if end_idx - start_idx < context_size and start_idx > 0:
            start_idx = max(0, end_idx - context_size)

        window = list(range(start_idx, end_idx))
        windows.append(window)

    # Record frame mappings for debugging
    frame_map = {}
    for i, window in enumerate(windows):
        for frame in window:
            if frame not in frame_map:
                frame_map[frame] = []
            frame_map[frame].append(i)

    # Log frames that appear in multiple windows (potential overlap issues)
    multi_window_frames = {f: ws for f, ws in frame_map.items() if len(ws) > 1}
    if multi_window_frames:
        log.info(f"Found {len(multi_window_frames)} frames in multiple windows")

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
