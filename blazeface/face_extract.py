import os
from typing import Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

from blazeface import BlazeFace


class FaceExtractor:
    def __init__(self, video_read_fn = None, facedet: BlazeFace = None):
        self.video_read_fn = video_read_fn
        self.facedet = facedet

    def process_image(self, path: str = None, img: Image.Image or np.ndarray = None) -> dict:
        if img is not None and path is not None:
            raise ValueError('Only one argument between path and img can be specified')
        if img is None and path is None:
            raise ValueError('At least one argument between path and img must be specified')

        target_size = self.facedet.input_size

        if img is None:
            img = np.asarray(Image.open(str(path)))
        else:
            img = np.asarray(img)

        tiles, resize_info = self._tile_frames(np.expand_dims(img, 0), target_size)
        detections = self.facedet.predict_on_batch(tiles, apply_nms=False)

        detections = self._resize_detections(detections, target_size, resize_info)

        num_frames = 1
        frame_size = (img.shape[1], img.shape[0])
        detections = self._untile_detections(num_frames, frame_size, detections)

        detections = self.facedet.nms(detections)

        frameref_detections = self._add_margin_to_detections(detections[0], frame_size, 0.2)
        faces = self._crop_faces(img, frameref_detections)
        kpts = self._crop_kpts(img, detections[0], 0.3)

        # Add additional information about the frame and detections.
        scores = list(detections[0][:, 16].cpu().numpy())
        frame_dict = {"frame_w": frame_size[0],
                      "frame_h": frame_size[1],
                      "faces": faces,
                      "kpts": kpts,
                      "detections": frameref_detections.cpu().numpy(),
                      "scores": scores,
                      }

        # Sort faces by descending confidence
        frame_dict = self._soft_faces_by_descending_score(frame_dict)

        return frame_dict

    def _soft_faces_by_descending_score(self, frame_dict: dict) -> dict:
        if len(frame_dict['scores']) > 1:
            sort_idxs = np.argsort(frame_dict['scores'])[::-1]
            new_faces = [frame_dict['faces'][i] for i in sort_idxs]
            new_kpts = [frame_dict['kpts'][i] for i in sort_idxs]
            new_detections = frame_dict['detections'][sort_idxs]
            new_scores = [frame_dict['scores'][i] for i in sort_idxs]
            frame_dict['faces'] = new_faces
            frame_dict['kpts'] = new_kpts
            frame_dict['detections'] = new_detections
            frame_dict['scores'] = new_scores
        return frame_dict

    def process_videos(self, input_dir, filenames, video_idxs) -> List[dict]:
        target_size = self.facedet.input_size

        videos_read = []
        frames_read = []
        frames = []
        tiles = []
        resize_info = []

        for video_idx in video_idxs:
            # Read the full-size frames from this video.
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)

            # Error? Then skip this video.
            if result is None: continue

            videos_read.append(video_idx)

            # Keep track of the original frames (need them later).
            my_frames, my_idxs = result
            frames.append(my_frames)
            frames_read.append(my_idxs)

            # Split the frames into several tiles. Resize the tiles to 128x128.
            my_tiles, my_resize_info = self._tile_frames(my_frames, target_size)
            tiles.append(my_tiles)
            resize_info.append(my_resize_info)

        if len(tiles) == 0:
            return []
        # Put all the tiles for all the frames from all the videos into
        # a single batch.
        batch = np.concatenate(tiles)

        # Run the face detector. The result is a list of PyTorch tensors,
        # one for each image in the batch.
        all_detections = self.facedet.predict_on_batch(batch, apply_nms=False)

        result = []
        offs = 0
        for v in range(len(tiles)):
            num_tiles = tiles[v].shape[0]
            detections = all_detections[offs:offs + num_tiles]
            offs += num_tiles

            detections = self._resize_detections(detections, target_size, resize_info[v])

            num_frames = frames[v].shape[0]
            frame_size = (frames[v].shape[2], frames[v].shape[1])
            detections = self._untile_detections(num_frames, frame_size, detections)

            detections = self.facedet.nms(detections)

            for i in range(len(detections)):
                frameref_detections = self._add_margin_to_detections(detections[i], frame_size, 0.2)
                faces = self._crop_faces(frames[v][i], frameref_detections)
                kpts = self._crop_kpts(frames[v][i], detections[i], 0.3)

                scores = list(detections[i][:, 16].cpu().numpy())
                frame_dict = {"video_idx": videos_read[v],
                              "frame_idx": frames_read[v][i],
                              "frame_w": frame_size[0],
                              "frame_h": frame_size[1],
                              "frame": frames[v][i],
                              "faces": faces,
                              "kpts": kpts,
                              "detections": frameref_detections.cpu().numpy(),
                              "scores": scores,
                              }
                frame_dict = self._soft_faces_by_descending_score(frame_dict)

                result.append(frame_dict)

        return result

    def process_video(self, video_path):
        input_dir = os.path.dirname(video_path)
        filenames = [os.path.basename(video_path)]
        return self.process_videos(input_dir, filenames, [0])

    def _tile_frames(self, frames: np.ndarray, target_size: Tuple[int, int]) -> (np.ndarray, List[float]):
        num_frames, H, W, _ = frames.shape

        num_h, num_v, split_size, x_step, y_step = self.get_tiles_params(H, W)

        splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)

        i = 0
        for f in range(num_frames):
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    crop = frames[f, y:y + split_size, x:x + split_size, :]
                    splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step

        resize_info = [split_size / target_size[0], split_size / target_size[1], 0, 0]
        return splits, resize_info

    def get_tiles_params(self, H, W):
        split_size = min(H, W, 720)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = (H - split_size) // y_step + 1 if y_step > 0 else 1
        num_h = (W - split_size) // x_step + 1 if x_step > 0 else 1
        return num_h, num_v, split_size, x_step, y_step

    def _resize_detections(self, detections, target_size, resize_info):
        projected = []
        target_w, target_h = target_size
        scale_w, scale_h, offset_x, offset_y = resize_info

        for i in range(len(detections)):
            detection = detections[i].clone()

            # ymin, xmin, ymax, xmax
            for k in range(2):
                detection[:, k * 2] = (detection[:, k * 2] * target_h - offset_y) * scale_h
                detection[:, k * 2 + 1] = (detection[:, k * 2 + 1] * target_w - offset_x) * scale_w

            # keypoints are x,y
            for k in range(2, 8):
                detection[:, k * 2] = (detection[:, k * 2] * target_w - offset_x) * scale_w
                detection[:, k * 2 + 1] = (detection[:, k * 2 + 1] * target_h - offset_y) * scale_h

            projected.append(detection)

        return projected

    def _untile_detections(self, num_frames: int, frame_size: Tuple[int, int], detections: List[torch.Tensor]) -> List[
        torch.Tensor]:
        combined_detections = []

        W, H = frame_size

        num_h, num_v, split_size, x_step, y_step = self.get_tiles_params(H, W)

        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    # Adjust the coordinates based on the split positions.
                    detection = detections[i].clone()
                    if detection.shape[0] > 0:
                        for k in range(2):
                            detection[:, k * 2] += y
                            detection[:, k * 2 + 1] += x
                        for k in range(2, 8):
                            detection[:, k * 2] += x
                            detection[:, k * 2 + 1] += y

                    detections_for_frame.append(detection)
                    x += x_step
                    i += 1
                y += y_step

            combined_detections.append(torch.cat(detections_for_frame))

        return combined_detections

    def _add_margin_to_detections(self, detections: torch.Tensor, frame_size: Tuple[int, int],
                                  margin: float = 0.2) -> torch.Tensor:
        offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset * 2, min=0)  # ymin
        detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)  # xmin
        detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  # ymax
        detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  # xmax
        return detections

    def _crop_faces(self, frame: np.ndarray, detections: torch.Tensor) -> List[np.ndarray]:
        faces = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(int)
            face = frame[ymin:ymax, xmin:xmax, :]
            faces.append(face)
        return faces

    def _crop_kpts(self, frame: np.ndarray, detections: torch.Tensor, face_fraction: float):
        faces = []
        for i in range(len(detections)):
            kpts = []
            size = int(face_fraction * min(detections[i, 2] - detections[i, 0], detections[i, 3] - detections[i, 1]))
            kpts_coords = detections[i, 4:16].cpu().numpy().astype(int)
            for kpidx in range(6):
                kpx, kpy = kpts_coords[kpidx * 2:kpidx * 2 + 2]
                kpt = frame[kpy - size // 2:kpy - size // 2 + size, kpx - size // 2:kpx - size // 2 + size, ]
                kpts.append(kpt)
            faces.append(kpts)
        return faces

    def remove_large_crops(self, crops, pct=0.1):
        for i in range(len(crops)):
            frame_data = crops[i]
            video_area = frame_data["frame_w"] * frame_data["frame_h"]
            faces = frame_data["faces"]
            scores = frame_data["scores"]
            new_faces = []
            new_scores = []
            for j in range(len(faces)):
                face = faces[j]
                face_H, face_W, _ = face.shape
                face_area = face_H * face_W
                if face_area / video_area < 0.1:
                    new_faces.append(face)
                    new_scores.append(scores[j])
            frame_data["faces"] = new_faces
            frame_data["scores"] = new_scores

    def keep_only_best_face(self, crops):
        for i in range(len(crops)):
            frame_data = crops[i]
            if len(frame_data["faces"]) > 0:
                frame_data["faces"] = frame_data["faces"][:1]
                frame_data["scores"] = frame_data["scores"][:1]

