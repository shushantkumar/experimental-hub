"""Provide `RPPGHeartRateFilter` filter."""

import numpy
from .rppg_utils import RPPG
from .rppg_utils.processors import ColorMeanProcessor, FilteredProcessor
from .rppg_utils.hr import HRCalculator
from .rppg_utils.filters import get_butterworth_filter

from .rppg_utils.roi.roi_detect import FaceMeshDetector
from .rppg_utils.processors.li_cvpr import LiCvprProcessor
from av import VideoFrame

from filters.filter import Filter
from filters import FilterDict


class RPPGHeartRateFilter(Filter):
    """Filter to calculate Heart Rate using RPPG algorithm
    """
    def __init__(self, config: FilterDict, audio_track_handler, video_track_handler):
        super().__init__(config, audio_track_handler, video_track_handler)
        self.roi_detector = FaceMeshDetector(draw_landmarks=True)

        digital_lowpass = get_butterworth_filter(30, 1.5)
        self.hr_calc = HRCalculator(update_interval=30, winsize=300, filt_fun=lambda vs: [digital_lowpass(v) for v in vs])

        cutoff="0.5,2"
        bandpass_cutoff = list(map(float, cutoff.split(",")))
        digital_bandpass = get_butterworth_filter(30, bandpass_cutoff, "bandpass")
        self.processor = LiCvprProcessor()
        self.processor = FilteredProcessor(self.processor, digital_bandpass)
        self.rppg = RPPG(roi_detector=self.roi_detector, hr_calculator=self.hr_calc)
        self.rppg.add_processor(self.processor)
        for c in "rgb":
            self.rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1))


    @staticmethod
    def name(self) -> str:
        return "RPPG_HEART_RATE"

    async def process(self, _: VideoFrame, ndarray: numpy.ndarray) -> numpy.ndarray:
        ndarray = self.rppg.on_frame_received(ndarray)
        return ndarray
