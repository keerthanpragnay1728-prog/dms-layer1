"""Layer 1 of the driver monitoring system: facial landmark perception.

Face detection (Haar) + a 24-point landmark model, with the landmark source
swappable behind one interface so our model and MediaPipe can be ablated
against an identical downstream pipeline.
"""

__version__ = "0.1.0"
