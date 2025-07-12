import cv2
import numpy as np


class VideoReader:
    def __init__(self, verbose=True, insets=(0, 0)):
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path, num_frames, jitter=0, seed=None):
        assert num_frames > 0

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int)
        frame_idxs = np.unique(frame_idxs)
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

        result = self._read_frames_at_indices(capture, path, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, capture, path, frame_idxs):
        try:
            frames = []
            idxs_read = []
            current_idx = 0

            for i in range(frame_idxs[0], frame_idxs[-1] + 1):
                ret = capture.grab()
                if not ret:
                    if self.verbose:
                        print(f"Error grabbing frame {i} from {path}")
                    break

                if i == frame_idxs[current_idx]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            print(f"Error retrieving frame {i} from {path}")
                        break
                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(i)
                    current_idx += 1
                    if current_idx >= len(frame_idxs):
                        break

            if len(frames) > 0:
                return np.stack(frames), idxs_read

            if self.verbose:
                print(f"No frames read from {path}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Exception while reading {path}: {e}")
            return None

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]

        if self.insets[1] > 0:
            H = frame.shape[0]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]

        return frame
