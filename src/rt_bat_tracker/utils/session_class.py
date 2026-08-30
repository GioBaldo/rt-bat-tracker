from rt_bat_tracker.utils.event_class import Event
import time
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Session:
    """
    Session class is istantiated only once for the time the bat tracker is running and it will contain
    all the events detected.
    It must store general informations about the session like time, config file used, number of events detected, ecc.
    This must also handle the birth and death of new events checking the sleep_timer when no new calls are stored
    The session will be saved as a csv file in order to be able to acces each event any moment also after shuting down the bat-tracker
    """

    def __init__(self, cfg, state, projPaths, name=time.asctime()):
        self._state = state
        self._cfg = cfg
        self.results_path = projPaths.results_dir
        self.event_list: list[Event] = []
        self.active_event = None
        self.session_name = name
        self.start_time = time.monotonic()

        self.wav_path = Path(self.results_path / self.session_name)

    def new_event(self, timestamp):
        event = Event(str(len(self.event_list) + 1), timestamp)
        self.event_list.append(event)
        self.active_event = event
        self._state.empty_res_count = 0
        return self.active_event

    def update(self, pos, timestamp):
        """handles session update:
        if there is no active event creates one
        if the active event is on for more than the max_event_time given replaces it with a new one
        if no data on input keeps the event alive only for evetn_max_sleep_time seconds and then kills the event

        Args:
            pos (new point position or None): already validated results [x,y,z] or None
            timestamp (adc timestamp): adc synchronized timestamp
        """
        logger.debug(f"results positions is passed to the session as: {pos}")
        if pos is None:
            if self.active_event is not None:
                if (
                    time.monotonic() - self.active_event.last_call_time
                    > self._cfg.event_max_sleep_time
                ):  # an event is active but no new point are coming in after event_max_sleep_time seconds, so we kill the event
                    self.kill_event()
        else:

            if self.active_event is None:
                # new points are coming in but no event is active, so we create a new one
                self.new_event(timestamp)
            else:
                if (
                    time.monotonic() - self.active_event.start_time
                    > self._cfg.max_event_time
                ):  # an event is active but it has been running for more than max_event_time seconds, so we kill the event and create a new one
                    self.kill_event()
                    self.new_event(timestamp)

            # add the new point to the active event
            self.active_event.add_point(pos, timestamp)
            logger.info(
                f"Point added to {self.active_event.event_name}, points stored: {len(self.active_event.points)}"
            )
            self.active_event.last_call_time = time.monotonic()

    def read_event(self, this_event):
        # logger.debug(
        #     f"session read requested... active event = None? ({this_event == None})"
        # )
        if (
            this_event is not None
        ):  # this may not work with passed events, need to be adapted
            active_points = []
            ap_colors = []
            all_times = []
            for p in this_event.points:
                age = (
                    time.monotonic() - p.rel_ts - this_event.start_time
                )  # this should ensure a timewise consistent plot
                # logger.debug(
                #     f"age: {age} : [ {time.monotonic()} + {this_event.start_time_adc} - {p.rel_ts}]"
                # )
                if age > 0:
                    op = (
                        max(0, 1 - age / self._state.fade_time)
                        if self._state.fade
                        else 1
                    )
                    p.color = np.append(self._state.tail_color, op)

                    active_points.append(p.pos)
                    ap_colors.append(p.color)

                    all_times.append(p.rel_ts)
                    # not used now but maybe convert to realtive_ts
            logger.debug(
                f"session is returning points: {np.shape(active_points)}, and colors: {np.shape(ap_colors)}"
            )
            return active_points, ap_colors, all_times
        else:
            return None, None, None

    def kill_event(self):
        """killing an event means placing it in the event list complete of the entire spectrogram
        of one track, the audio file? some stats about the event?
        """
        # WILL IMPLEMENT ALSO SPECTROGRAM STORE AND OTHER DATA

        if self.active_event is not None:
            logger.info(f"killing event: {self.active_event.event_name}")
            self.active_event.terminate_event()
            if self._cfg.SAVE_RESULTS:
                self.wav_path.mkdir()
                self.save_wav_file()
            self.active_event = None
        else:
            logger.info("No active events to be killed")

    def write_audiofile(self):
        """if an event is active stores the waveforms of all channels to be saved in a file at the oend of the event
        [Storing also the timestamp could be useful for further analysis, like identifying the exact point related to the
        call in the audiofile. Will implement this later]
        When new audio data are collected, they are processed via fft to obtain tha spectrogram, also stored in the event
        """
        logger.debug(f"writing audiofile to active event {self.active_event} ")
        if self.active_event is not None:
            data = self._state.grab_wav_buffer()

            logger.debug(
                f"about to append data: ({np.shape(data)[0]}) [{type(data)}] to the audio_file: [{np.shape(self.active_event.audio_file)}]"
            )
            self.update_spectrogram(data, 0)  # performing spectrogram on ch 0 only

            for i in range(np.shape(data)[0]):
                self.active_event.audio_file.append(data[i])

    def save_wav_file(self):

        path = self.wav_path / f"{self.active_event.event_name}.wav"

        logger.info(
            f"saving audio file for {self.active_event.event_name} at {path} with shape {np.shape(self.active_event.audio_file)}"
        )
        sf.write(path, np.vstack(self.active_event.audio_file), self._cfg.fs)

    def update_spectrogram(self, data, ch):
        """
        perform fft on one channel and save the (samples, bins) np.array of np.float16 in the event spectrogram
        """
        taim = time.perf_counter_ns()
        data = np.array(data)
        signal = data[:, :, ch].reshape(-1)
        NFFT = self._cfg.WINDOW_SIZE
        HOP = self._cfg.HOP_SIZE
        window = np.hanning(NFFT)
        logger.debug(
            f"performing fft on data {np.shape(signal)}, NFFT[{NFFT}], HOPSIZE[{HOP}] "
        )
        for start in range(0, len(signal) - NFFT, HOP):
            frame = signal[start : start + NFFT]
            frame = frame * window
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum).astype(np.float16)
            # magnitude_db = 20 * np.log10(magnitude, 1e-12)
            self.active_event.spectrogram.append(magnitude)
        logger.debug(
            f"updated fft: actual size [{np.shape(self.active_event.spectrogram)} type {type(magnitude[1])} took {time.perf_counter_ns() - taim} nanosec]"
        )

    def read_spectrogram_for_playback(self, event, playback_time, (bins, num_frames)):
        """
        Given the event and the time in seconds this fucntion returns the spectrogram of the event before the playback_time
        """
        if event is not None:
            # Calculate the number of frames to include based on playback_time
            frame_idx = int(playback_time * self._cfg.fs / self._cfg.HOP_SIZE)
            frame_idx = min(frame_idx, len(event.spectrogram))
            num_frames = min(num_frames, frame_idx)
            array = np.zeros((num_frames, bins), dtype=np.uint8)
            data = np.array(event.spectrogram[frame_idx - num_frames:frame_idx], dtype=np.uint8)
            array[:data.shape[0], :data.shape[1]] = data
            return array
        else:
            return np.array([])
