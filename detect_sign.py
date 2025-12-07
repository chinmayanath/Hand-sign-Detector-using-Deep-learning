"""
Real-Time Sign Language Detection System

This module provides real-time sign language gesture recognition using a webcam feed.
It processes video frames, extracts skeletal keypoints using MediaPipe Holistic,
and classifies gestures using a pre-trained LSTM model.

Features:
- Real-time gesture detection with confidence thresholding
- Visual feedback with prediction confidence bars
- Frame rate monitoring and performance statistics
- Configurable detection parameters
- Robust error handling and resource management

Author: Refactored version
Date: 2025-10-20
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Deque
from dataclasses import dataclass
from collections import deque
import time

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model, Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration parameters for sign language detection."""
    
    # Model configuration
    model_path: Path
    actions: List[str]
    sequence_length: int = 30
    
    # Detection parameters
    confidence_threshold: float = 0.8
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Camera configuration
    camera_index: int = 0
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    
    # Display configuration
    show_landmarks: bool = True
    show_fps: bool = True
    show_confidence_bar: bool = True
    
    # Keypoint dimensions (must match training data)
    POSE_LANDMARKS: int = 33
    POSE_FEATURES: int = 4
    FACE_LANDMARKS: int = 468
    FACE_FEATURES: int = 3
    HAND_LANDMARKS: int = 21
    HAND_FEATURES: int = 3
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")


class KeypointExtractor:
    """Extracts and processes skeletal keypoints from MediaPipe Holistic results."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._pose_size = config.POSE_LANDMARKS * config.POSE_FEATURES
        self._face_size = config.FACE_LANDMARKS * config.FACE_FEATURES
        self._hand_size = config.HAND_LANDMARKS * config.HAND_FEATURES
        self.total_features = self._pose_size + self._face_size + 2 * self._hand_size
    
    def extract(self, results) -> np.ndarray:
        """
        Extract keypoints from MediaPipe Holistic results.
        
        Args:
            results: MediaPipe Holistic detection results
            
        Returns:
            Flattened numpy array of shape (total_features,)
        """
        pose = self._extract_pose(results)
        face = self._extract_face(results)
        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)
        
        return np.concatenate([pose, face, left_hand, right_hand])
    
    def _extract_pose(self, results) -> np.ndarray:
        """Extract pose landmarks with visibility."""
        if results.pose_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()
        return np.zeros(self._pose_size, dtype=np.float32)
    
    def _extract_face(self, results) -> np.ndarray:
        """Extract face landmarks."""
        if results.face_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ]).flatten()
        return np.zeros(self._face_size, dtype=np.float32)
    
    def _extract_hand(self, hand_landmarks) -> np.ndarray:
        """Extract hand landmarks."""
        if hand_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ]).flatten()
        return np.zeros(self._hand_size, dtype=np.float32)


class PredictionSmoother:
    """Smooths predictions over time to reduce flickering."""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize prediction smoother.
        
        Args:
            window_size: Number of recent predictions to consider
        """
        self.window_size = window_size
        self.prediction_history: Deque[Tuple[str, float]] = deque(maxlen=window_size)
    
    def add_prediction(self, action: str, confidence: float) -> None:
        """Add a new prediction to the history."""
        self.prediction_history.append((action, confidence))
    
    def get_smoothed_prediction(self) -> Optional[Tuple[str, float]]:
        """
        Get the most common prediction with average confidence.
        
        Returns:
            Tuple of (action, average_confidence) or None if no predictions
        """
        if not self.prediction_history:
            return None
        
        # Count occurrences of each action
        action_votes = {}
        for action, confidence in self.prediction_history:
            if action not in action_votes:
                action_votes[action] = []
            action_votes[action].append(confidence)
        
        # Find most common action
        best_action = max(action_votes.keys(), key=lambda k: len(action_votes[k]))
        avg_confidence = np.mean(action_votes[best_action])
        
        return best_action, avg_confidence
    
    def reset(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames to average for FPS calculation
        """
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.last_time = time.time()
        self.frame_count = 0
        self.prediction_count = 0
    
    def update(self) -> None:
        """Update with new frame timing."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        self.frame_count += 1
    
    def get_fps(self) -> float:
        """Calculate average FPS."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def record_prediction(self) -> None:
        """Record a successful prediction."""
        self.prediction_count += 1
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'fps': self.get_fps(),
            'total_frames': self.frame_count,
            'total_predictions': self.prediction_count,
            'avg_frame_time_ms': np.mean(self.frame_times) * 1000 if self.frame_times else 0
        }


class SignLanguageDetector:
    """Main class for real-time sign language detection."""
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize the detector.
        
        Args:
            config: Detection configuration object
        """
        self.config = config
        self.model = self._load_model()
        self.extractor = KeypointExtractor(config)
        self.smoother = PredictionSmoother(window_size=5)
        self.performance = PerformanceMonitor()
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Sequence buffer
        self.sequence: Deque[np.ndarray] = deque(maxlen=config.sequence_length)
        
        # State tracking
        self.current_prediction: Optional[str] = None
        self.current_confidence: float = 0.0
        
        logger.info(f"Detector initialized with model: {config.model_path.name}")
        logger.info(f"Actions: {', '.join(config.actions)}")
    
    def _load_model(self) -> Model:
        """Load the trained model."""
        try:
            model = load_model(str(self.config.model_path))
            logger.info(f"Model loaded successfully: {self.config.model_path}")
            logger.info(f"Model input shape: {model.input_shape}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
        
        # Set resolution if specified
        if self.config.frame_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        if self.config.frame_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        
        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera initialized: {width}x{height}")
        
        return cap
    
    def _process_frame(self, frame: np.ndarray, results) -> None:
        """
        Process a single frame and make predictions.
        
        Args:
            frame: Input video frame
            results: MediaPipe Holistic results
        """
        # Extract keypoints
        keypoints = self.extractor.extract(results)
        self.sequence.append(keypoints)
        
        # Make prediction when sequence is full
        if len(self.sequence) == self.config.sequence_length:
            sequence_array = np.expand_dims(self.sequence, axis=0)
            predictions = self.model.predict(sequence_array, verbose=0)[0]
            
            max_confidence = float(np.max(predictions))
            predicted_idx = int(np.argmax(predictions))
            
            # Check confidence threshold
            if max_confidence > self.config.confidence_threshold:
                predicted_action = self.config.actions[predicted_idx]
                self.smoother.add_prediction(predicted_action, max_confidence)
                
                # Get smoothed prediction
                smoothed = self.smoother.get_smoothed_prediction()
                if smoothed:
                    self.current_prediction, self.current_confidence = smoothed
                    self.performance.record_prediction()
            else:
                self.current_prediction = None
                self.current_confidence = 0.0
    
    def _draw_landmarks(self, frame: np.ndarray, results) -> None:
        """Draw MediaPipe landmarks on the frame."""
        if not self.config.show_landmarks:
            return
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
    
    def _draw_ui(self, frame: np.ndarray) -> None:
        """Draw UI elements on the frame."""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Draw prediction
        if self.current_prediction:
            # Main prediction box
            box_height = 120
            cv2.rectangle(overlay, (10, 10), (width - 10, box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Prediction text
            cv2.putText(
                frame,
                f"Sign: {self.current_prediction.upper()}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
            
            # Confidence bar
            if self.config.show_confidence_bar:
                self._draw_confidence_bar(frame, 20, 70, width - 40, 30)
        
        # Draw FPS
        if self.config.show_fps:
            fps = self.performance.get_fps()
            fps_text = f"FPS: {fps:.1f}"
            
            # FPS background
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (width - text_size[0] - 20, height - 40),
                (width - 10, height - 10),
                (0, 0, 0),
                -1
            )
            
            # FPS text
            color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
            cv2.putText(
                frame,
                fps_text,
                (width - text_size[0] - 15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
        
        # Draw instructions
        instructions = [
            "Press 'Q' to quit",
            "Press 'R' to reset",
            "Press 'S' for stats"
        ]
        y_offset = height - 100
        for instruction in instructions:
            cv2.putText(
                frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 20
    
    def _draw_confidence_bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> None:
        """Draw a confidence level bar."""
        # Background bar
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Confidence bar
        confidence_width = int(width * self.current_confidence)
        color = self._get_confidence_color(self.current_confidence)
        cv2.rectangle(frame, (x, y), (x + confidence_width, y + height), color, -1)
        
        # Confidence text
        conf_text = f"{self.current_confidence * 100:.1f}%"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        
        cv2.putText(
            frame,
            conf_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level (BGR format)."""
        if confidence > 0.9:
            return (0, 255, 0)  # Green
        elif confidence > 0.8:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 165, 255)  # Orange
    
    def _print_statistics(self) -> None:
        """Print performance statistics to console."""
        stats = self.performance.get_stats()
        logger.info("\n" + "="*50)
        logger.info("Performance Statistics:")
        logger.info("-"*50)
        logger.info(f"  Average FPS: {stats['fps']:.2f}")
        logger.info(f"  Total Frames: {stats['total_frames']}")
        logger.info(f"  Total Predictions: {stats['total_predictions']}")
        logger.info(f"  Avg Frame Time: {stats['avg_frame_time_ms']:.2f} ms")
        if stats['total_frames'] > 0:
            pred_rate = stats['total_predictions'] / stats['total_frames'] * 100
            logger.info(f"  Prediction Rate: {pred_rate:.1f}%")
        logger.info("="*50 + "\n")
    
    def run(self) -> None:
        """
        Run the real-time detection system.
        
        Main loop that captures video, processes frames, and displays results.
        """
        cap = self._initialize_camera()
        if cap is None:
            logger.error("Cannot initialize camera")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Starting real-time sign language detection...")
        logger.info("="*60)
        logger.info("Controls:")
        logger.info("  Q - Quit")
        logger.info("  R - Reset sequence buffer")
        logger.info("  S - Show statistics")
        logger.info("="*60 + "\n")
        
        try:
            with self.mp_holistic.Holistic(
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                model_complexity=1  # Balance between speed and accuracy
            ) as holistic:
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.error("Failed to read frame from camera")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame with MediaPipe
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = holistic.process(image_rgb)
                    image_rgb.flags.writeable = True
                    
                    # Make predictions
                    self._process_frame(frame, results)
                    
                    # Draw landmarks
                    self._draw_landmarks(frame, results)
                    
                    # Draw UI
                    self._draw_ui(frame)
                    
                    # Update performance monitor
                    self.performance.update()
                    
                    # Display frame
                    cv2.imshow('Sign Language Detection', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        logger.info("Quit command received")
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.sequence.clear()
                        self.smoother.reset()
                        logger.info("Sequence buffer reset")
                    elif key == ord('s') or key == ord('S'):
                        self._print_statistics()
        
        except KeyboardInterrupt:
            logger.info("\nDetection interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_statistics()
            
            logger.info("Detection system stopped")


def main():
    """Main entry point for the detection script."""
    
    # Configuration
    config = DetectionConfig(
        model_path=Path('models/sign_model.h5'),
        actions=["hello", "thanks", "iloveyou"],
        sequence_length=30,
        confidence_threshold=0.8,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        camera_index=0,
        show_landmarks=True,
        show_fps=True,
        show_confidence_bar=True
    )
    
    try:
        # Initialize and run detector
        detector = SignLanguageDetector(config)
        detector.run()
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())