from rt_bat_tracker.utils.event_class import Event
import time
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
import csv
import json

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

    def __init__(self, cfg, state, projPaths, name=None):
        self._state = state
        self._cfg = cfg
        self.results_path = projPaths.results_dir
        self.event_list: list[Event] = []
        self.active_event = None
        self.session_name = name
        if self.session_name is None:
            self.session_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        else:
            self.session_name = name
        self.start_time = time.monotonic()
        self.session_path = Path(self.results_path / self.session_name)

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
                ) 
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
            if self._state.SAVE_RESULTS:
                self._save_event(self.active_event)
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
            magnitude = np.abs(spectrum*100.0).astype(np.float16)
            # magnitude_db = 20 * np.log10(magnitude, 1e-12)
            self.active_event.spectrogram.append(magnitude)
        logger.debug(
            f"updated fft: actual size [{np.shape(self.active_event.spectrogram)} type {type(magnitude[1])} took {time.perf_counter_ns() - taim} nanosec]"
        )

    def read_points_for_playback(self, this_event, playback_time):
        """
        Given the event and the time in seconds this fucntion returns the points of the event before the playback_time
        """
        if (
            this_event is not None
        ):  # this may not work with passed events, need to be adapted
            active_points = []
            ap_colors = []
            for p in this_event.points:
                age = (playback_time - p.rel_ts) 
                if age > 0:
                    op = (
                        max(0, 1 - age / self._state.fade_time)
                        if self._state.fade
                        else 1
                    )
                    p.color = np.append(self._state.tail_color, op)
                    active_points.append(p.pos)
                    ap_colors.append(p.color)
            
            return active_points, ap_colors
        else:
            return None, None, None

    def read_spectrogram_for_playback(self, event, playback_time, shape):
        """
        Given the event and the time in seconds this fucntion returns the spectrogram of the event before the playback_time
        """
        bins, num_frames = shape
        if event is not None:
            # Calculate the number of frames to include based on playback_time
            frame_idx = int(playback_time * self._cfg.fs / self._cfg.HOP_SIZE)
            frame_idx = min(frame_idx, len(event.spectrogram))
            data_frames = min(num_frames, frame_idx)
            array = np.zeros((bins, num_frames), dtype=np.uint8)
            logger.debug(f"Array shape: {array.shape}, spectrogram length: {len(event.spectrogram)}, {len(event.spectrogram[0])}")
            data = np.array(event.spectrogram[frame_idx - data_frames:frame_idx], dtype=np.uint8)
            array[:, :data_frames] = data.T
            return array
        else:
            return np.array([])

    def _save_event(self, target_event):
        """
        Save one event data to disk:
        - WAV audio file
        - points as CSV
        - spectrogram as NPY
        - event metadata in session manifest JSON
        """
        self.session_path.mkdir(exist_ok=True, parents=True)
        event_stem = target_event.event_name.replace(" ", "_")
        self.event_path = self.session_path / event_stem
        self.event_path.mkdir(exist_ok=True, parents=True)

        ## SAVING WAV FILE ##
        wav_filename = f"{event_stem}.wav"
        wav_path = self.event_path / wav_filename

        logger.info(
            f"saving audio file for {target_event.event_name} at {wav_path} with shape {np.shape(target_event.audio_file)}"
        )
        sf.write(wav_path, np.vstack(target_event.audio_file), self._cfg.fs)

        ## SAVING POINTS ##
        points_filename = f"{event_stem}.points.csv"
        points_path = self.event_path / points_filename

        with open(points_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["abs_ts", "rel_ts", "x", "y", "z"])
            for p in target_event.points:
                writer.writerow(
                    [
                        float(p.abs_ts),
                        float(p.rel_ts),
                        float(p.pos[0]),
                        float(p.pos[1]),
                        float(p.pos[2]),
                    ]
                )

        ## SAVING SPECTROGRAM ##
        spectrogram_filename = f"{event_stem}.spectrogram.npy"
        spectrogram_path = self.event_path / spectrogram_filename
        np.save(spectrogram_path, target_event.spectrogram)

        ## SAVING METADATA ##
        manifest_path = self.session_path / "session_manifest.json"
        
        #initialize manifest in case it doesnt exist yet
        manifest = {
            "session_name": self.session_name,
            "events": []}

        #if the manifest exist loads the existing one
        if manifest_path.exists():
            with open(manifest_path, mode="r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            manifest["session_name"] = self.session_name

        #creates the event record to be saved in the manifest
        event_record = {
            "event_name": target_event.event_name,
            "event_folder": event_stem,
            "start_time_adc": float(target_event.start_time_adc),
            "duration": (
                float(target_event.duration) if target_event.duration is not None else None
            ),
            "points_file": points_filename,
            "spectrogram_file": spectrogram_filename,
            "audio_file": wav_filename,
            "points_count": len(target_event.points),
            "spectrogram_frames": len(target_event.spectrogram),
        }

        #updates the manifest by removing any existing (shouldn't exixt) record for the same event name and appending the new record
        manifest["events"] = [
            e for e in manifest.get("events", []) if e.get("event_name") != target_event.event_name
        ]
        manifest["events"].append(event_record)

        with open(manifest_path, mode="w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)    

    def get_recall_session_list(self):
        """
        Returns a list of all sessions available in the results directory.
        Each session is represented by its name (directory name).
        """
        sessions = []
        for session_dir in self.results_path.iterdir():
            if session_dir.is_dir():
                sessions.append(session_dir.name)
        return sessions
        
    def get_recall_event_list(self, session_name):
        """
        Load events from disk and return a list of Event objects.
        """
        event_list = []
        session_path = self.results_path / session_name
        manifest_path = session_path / "session_manifest.json"
        if not manifest_path.exists():
            logger.warning(f"No manifest found at {manifest_path}. No events loaded.")
            return event_list

        with open(manifest_path, mode="r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        
        for event_index, event_data in enumerate(manifest.get("events", [])):
            event_name = event_data["event_name"]
            event_folder = event_data["event_folder"]
            start_time_adc = float(event_data["start_time_adc"])
            event = Event(event_index, start_time_adc)
            event.event_name = event_name
            event.duration = event_data.get("duration")
            event.last_call_time = time.monotonic()

            #load points and add to event
            points_path = session_path / event_folder / event_data["points_file"]
            with open(points_path, mode="r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    pos = np.array(
                        [float(row["x"]), float(row["y"]), float(row["z"])],
                        dtype=np.float32,
                    )
                    abs_ts = float(row["abs_ts"])
                    event.add_point(pos, abs_ts)
                    event.points[-1].rel_ts = float(row["rel_ts"])

            #load spectrogram        
            spectrogram_path = session_path / event_folder / event_data["spectrogram_file"]
            if spectrogram_path.exists():
                event.spectrogram = np.load(spectrogram_path).tolist()
            event_list.append(event)
            logger.info(f"Loaded event: {event_name} with {len(event.points)} points and {len(event.spectrogram)} spectrogram frames.")

        return event_list