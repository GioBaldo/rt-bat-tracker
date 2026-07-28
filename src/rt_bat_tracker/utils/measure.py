from rt_bat_tracker.audio.play_tone import PlayTone
import threading
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run(state, cfg, session):
    measure = Measure(state, cfg)
    state.gui_running_flag = True  # fake gui running
    measure.loop()
    state.stop(__name__)
    return


class Measure:
    def __init__(self, state, cfg):
        self._state = state
        self._cfg = cfg
        self.tone_frequency = 10000
        self.blocksize = 4096
        self.tone_rate = 1
        self.channels = [0, 1, 2]
        self.player = PlayTone(state, cfg)

    def loop(self):
        start_freq = 60000
        end_freq = 5000
        num_chunks = 3
        tone = self.player.make_sweep(start_freq, end_freq, num_chunks * self.blocksize)
        time.sleep(3)
        gnd_noise_avg = self._state.avg_rms
        gnd_noise_peak = self._state.max_rms
        logger.info(f"ground noise: avg={gnd_noise_avg}, peak={gnd_noise_peak}")
        for ch in self.channels:
            self.player.output(tone, ch, num_chunks, 1)
            time.sleep(1)

        source_rms_avg = self._state.avg_rms
        source_rms_peak = self._state.max_rms
        logger.info(f"source active: avg={source_rms_avg}, peak={source_rms_peak}")

        SNR_avg = source_rms_avg - gnd_noise_avg
        SNR_peak = source_rms_peak - gnd_noise_peak

        logger.info(f"Average SNR: {SNR_avg}, Peak SNR: {SNR_peak}")

        threshold = source_rms_peak * 0.8
        self._state.threshold = threshold
        logger.info(f"threshold set to {threshold}")

        # while True:
        #     result = self._state.get_result()
        #     if result is None:
        #         break

        logger.info(f"result queue is empty [len = {self._state.result_queue.qsize}]")
