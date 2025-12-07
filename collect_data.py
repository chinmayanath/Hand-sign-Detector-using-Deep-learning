"""
Sign Language Gesture Data Collector

This module captures video sequences and extracts skeletal keypoints for sign language
gesture recognition training data. It uses MediaPipe Holistic for body, face, and hand
landmark detection.

Author: Refactored version
Date: 2025-10-20
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataCollectionConfig:
    """Configuration parameters for data collection."""
    actions: List[str]
    data_path: Path
    num_sequences: int = 16
    frames_per_sequence: int = 16
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    camera_index: int = 0
    
    # Keypoint dimensions
    POSE_LANDMARKS: int = 33
    POSE_FEATURES: int = 4  # x, y, z, visibility
    FACE_LANDMARKS: int = 468
    FACE_FEATURES: int = 3  # x, y, z
    HAND_LANDMARKS: int = 21
    HAND_FEATURES: int = 3  # x, y, z


class KeypointExtractor:
    """Handles extraction of keypoints from MediaPipe Holistic results."""
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self._pose_size = config.POSE_LANDMARKS * config.POSE_FEATURES
        self._face_size = config.FACE_LANDMARKS * config.FACE_FEATURES
        self._hand_size = config.HAND_LANDMARKS * config.HAND_FEATURES
    
    def extract(self, results) -> np.ndarray:
        """
        Extract keypoints from MediaPipe Holistic results.
        
        Args:
            results: MediaPipe Holistic detection results
            
        Returns:
            Flattened numpy array containing all keypoints
        """
        pose = self._extract_pose(results)
        face = self._extract_face(results)
        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)
        
        return np.concatenate([pose, face, left_hand, right_hand])
    
    def _extract_pose(self, results) -> np.ndarray:
        """Extract pose landmarks (33 landmarks × 4 features)."""
        if results.pose_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()
        return np.zeros(self._pose_size)
    
    def _extract_face(self, results) -> np.ndarray:
        """Extract face landmarks (468 landmarks × 3 features)."""
        if results.face_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ]).flatten()
        return np.zeros(self._face_size)
    
    def _extract_hand(self, hand_landmarks) -> np.ndarray:
        """Extract hand landmarks (21 landmarks × 3 features)."""
        if hand_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ]).flatten()
        return np.zeros(self._hand_size)


class DataCollector:
    """Main class for collecting sign language gesture data."""
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.extractor = KeypointExtractor(config)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create necessary directories for data storage."""
        for action in self.config.actions:
            action_path = self.config.data_path / action
            action_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {action_path}")
    
    def _initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Initialize video capture device.
        
        Returns:
            VideoCapture object if successful, None otherwise
        """
        cap = cv2.VideoCapture(self.config.camera_index)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera at index {self.config.camera_index}")
            return None
        
        logger.info(f"Camera initialized successfully (index: {self.config.camera_index})")
        return cap
    
    def _collect_sequence(
        self,
        cap: cv2.VideoCapture,
        holistic,
        action: str,
        seq_num: int
    ) -> bool:
        """
        Collect a single sequence of frames for a specific action.
        
        Args:
            cap: VideoCapture object
            holistic: MediaPipe Holistic model
            action: Action name being recorded
            seq_num: Sequence number
            
        Returns:
            True if collection was successful, False if interrupted
        """
        seq_data = []
        logger.info(f"Recording '{action}' - Sequence {seq_num + 1}/{self.config.num_sequences}")
        
        for frame_num in range(self.config.frames_per_sequence):
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Failed to read frame {frame_num}")
                return False
            
            # Process frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Extract and store keypoints
            keypoints = self.extractor.extract(results)
            seq_data.append(keypoints)
            
            # Display frame with landmarks
            frame_display = self._draw_landmarks(frame, results)
            self._add_text_overlay(frame_display, action, seq_num, frame_num)
            cv2.imshow('Sign Language Data Collection', frame_display)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Collection interrupted by user")
                return False
        
        # Save sequence
        self._save_sequence(action, seq_num, seq_data)
        return True
    
    def _draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw detected landmarks on the frame."""
        frame_copy = frame.copy()
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_copy,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_copy,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_copy,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )
        
        return frame_copy
    
    def _add_text_overlay(
        self,
        frame: np.ndarray,
        action: str,
        seq_num: int,
        frame_num: int
    ) -> None:
        """Add informational text overlay to the frame."""
        height, width = frame.shape[:2]
        
        # Add semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add text information
        text_lines = [
            f"Action: {action.upper()}",
            f"Sequence: {seq_num + 1}/{self.config.num_sequences}",
            f"Frame: {frame_num + 1}/{self.config.frames_per_sequence}",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 25
    
    def _save_sequence(self, action: str, seq_num: int, seq_data: List[np.ndarray]) -> None:
        """Save sequence data to disk."""
        file_path = self.config.data_path / action / f"{seq_num}.npy"
        np.save(file_path, np.array(seq_data))
        logger.info(f"Saved: {file_path} (shape: {np.array(seq_data).shape})")
    
    def collect_all_data(self) -> bool:
        """
        Main method to collect all data for all actions.
        
        Returns:
            True if collection completed successfully, False otherwise
        """
        cap = self._initialize_camera()
        if cap is None:
            return False
        
        try:
            with self.mp_holistic.Holistic(
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            ) as holistic:
                
                for action in self.config.actions:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Starting collection for action: '{action}'")
                    logger.info(f"{'='*60}\n")
                    
                    for seq_num in range(self.config.num_sequences):
                        success = self._collect_sequence(cap, holistic, action, seq_num)
                        if not success:
                            logger.warning("Data collection interrupted")
                            return False
                    
                    logger.info(f"Completed collection for '{action}'\n")
                
                logger.info("\n" + "="*60)
                logger.info("Data collection completed successfully!")
                logger.info("="*60)
                return True
                
        except KeyboardInterrupt:
            logger.warning("\nCollection interrupted by user (Ctrl+C)")
            return False
        except Exception as e:
            logger.error(f"Error during data collection: {e}", exc_info=True)
            return False
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Resources released")


def main():
    """Main entry point for the data collection script."""
    # Configuration
    config = DataCollectionConfig(
        actions=["hello", "thanks", "iloveyou"],
        data_path=Path("dataset"),
        num_sequences=16,
        frames_per_sequence=16,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        camera_index=0
    )
    
    # Initialize and run collector
    collector = DataCollector(config)
    success = collector.collect_all_data()
    
    if success:
        logger.info("\nAll data collected successfully!")
        logger.info(f"Dataset location: {config.data_path.absolute()}")
    else:
        logger.error("\nData collection failed or was interrupted")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())