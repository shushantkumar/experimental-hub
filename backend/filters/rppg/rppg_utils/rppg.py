from collections import namedtuple
from datetime import datetime

import cv2
import math
import numpy as np

RppgResults = namedtuple("RppgResults", ["dt",
                                         "rawimg",
                                         "roi",
                                         "hr",
                                         "vs_iter",
                                         "ts",
                                         "fps",
                                         ])

class RPPG:
    def __init__(self, roi_detector, hr_calculator=None):
        self.roi = None
        self._processors = []
        self._roi_detector = roi_detector
        self._dts = []
        self.last_update = datetime.now()
        self.frame_counter = 0
        self.output_frame = None
        self.hr_calculator = hr_calculator

    def add_processor(self, processor):
        self._processors.append(processor)

    def on_frame_received(self, frame):
        self.frame_counter += 1
        self.output_frame = frame
        self.roi = self._roi_detector(frame)
        for processor in self._processors:
            processor(self.roi)

        if self.hr_calculator is not None:
            self.hr_calculator.update(self)

        dt = self._update_time()
        filtered_frame = self.rppg_updated(RppgResults(dt=dt, rawimg=frame, roi=self.roi,
                                       hr=self.hr_calculator.updated_hr, vs_iter=self.get_vs,
                                       ts=self.get_ts, fps=self.get_fps()))
        return filtered_frame

    def _update_time(self):
        dt = (datetime.now() - self.last_update).total_seconds()
        self.last_update = datetime.now()
        self._dts.append(dt)

        return dt

    def get_vs(self, n=None):
        for processor in self._processors:
            if n is None:
                yield np.array(processor.vs, copy=True)
            else:
                yield np.array(processor.vs[-n:], copy=True)

    def get_ts(self, n=None):
        if n is None:
            dts = self._dts
        else:
            dts = self._dts[-n:]
        return np.cumsum(dts)

    def get_fps(self, n=5):
        return 1/np.mean(self._dts[-n:])

    @property
    def num_processors(self):
        return len(self._processors)

    @property
    def processor_names(self):
        return [str(p) for p in self._processors]

    def rppg_updated(self, results):
        heart_rate = results.hr
        print(heart_rate)
        frame = results.rawimg
        if not math.isnan(heart_rate):
            cv2.putText(frame, f"Heart Rate: {int(heart_rate)} bpm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame
