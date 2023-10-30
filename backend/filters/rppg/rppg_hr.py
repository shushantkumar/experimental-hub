"""Provide `RPPGHeartRateFilter` filter."""

import cv2
import numpy
from av import VideoFrame

from filters.filter import Filter


class RPPGHeartRateFilter(Filter):
    """Filter to calculate Heart Rate using RPPG algorithm
    """

    @staticmethod
    def name(self) -> str:
        return "RPPG_HEART_RATE"

    async def process(self, _: VideoFrame, ndarray: numpy.ndarray) -> numpy.ndarray:
        ndarray = cv2.cvtColor(cv2.Canny(ndarray, 100, 200), cv2.COLOR_GRAY2BGR)
        return ndarray
