import alsaaudio as alsa
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AlsaAudioSource:

    """
        Captures live audio from a hardware device via Alsa in Raspian throug a focusrite Scarlett 18i20.
        Uses PyAlsaAudio library by @lassimmisch for bindings between alsa and python.

        PCM object needs to be configured with some values:
            PCM(type: int = PCM_PLAYBACK, mode: int = PCM_NORMAL, rate: int = 44100, channels: int = 2,
            format: int = PCM_FORMAT_S16_LE, periodsize: int = 32, periods: int = 4,
            device: str = 'default', cardindex: int = -1) -> PCM
        
        some of them can be change by the use and are given as inputs for this class, others are hard-coded in order
        to have the device properly working within the linux-alsa-focusrite fixed environment

        inputs:
            - queue
            - device
            - fs (rate)
            - channels
            - blocksize (periodsize)
                    

        this audiosource can just be started with start() and stopped with stop(). blocks are continuously pushed to the shared
        queue audio_queue as an np.array of shape (blocksize, channels) as <np.float32> samples

        the hardcoded values are 
        type: alsa.PCM_CAPTURE [1], the PCM is in CAPTURE mode for RECORDING (PLAYBACK otherwise)
        mode: alsa.PCM_NORMAL [0], the PCM is in NORMAL mode then BLOCKS the caller until a frame is full, nice to avoid empty reads 
                    (NONBLOCK otherwise)
        format: int = PCM_FORMAT_S32_LE actually the only sample format that works
        periods: int = 1, if more than one then a a larger number of blocks are passed each time
        cardindex: int = -1 NON USED
    """

    def __init__(self, state, device = 'hw:3,0', fs=192000, channels=8, blocksize=1024):
        self._state = state
        self.device = device #resolve_input_device(device)
        self.fs = fs
        self.channels = channels
        self.blocksize = blocksize
        self.PCM = None
        self.chunknum = 0
        self.timestamp = time.perf_counter_ns()
        

    def loop(self):
        """
        Block is red and data are pushed to the queue
        managed by te start function
        """
        if self.PCM.state() == 2:
            size, block = self.PCM.read()
            logger.debug(f"time elapsed for PCM initialization: {(time.perf_counter_ns() - self.timestamp)/100000}")
            self.timestamp = time.perf_counter_ns()

        elif self.PCM.state() == 3 :
                  
            size, block = self.PCM.read()

            if size < self.blocksize:
                return True
            else:
                xAr = np.frombuffer(block, dtype = np.int32)
                xAr = xAr.reshape(-1, 10)[:,:self.channels]
                xAr = xAr.astype(np.float32)/(2**31) 
                logger.debug(f"[{self.chunknum}] ({self.PCM.state()}) del: {(time.perf_counter_ns() - self.timestamp)/100000} x shape: {np.shape(xAr)} RMS: {self.compute_max(xAr)} ")
                self.timestamp = time.perf_counter_ns()
                self._state.put_audio(xAr,self.timestamp)
                self.chunknum +=1
        else:
            logger.info(f"LOOP RUNNING BUT PCM STATE IS {self.PCM.state()}")
            return True


    def start(self):
        """
        called by the audio_input.run() selector, this funcion initializes the PCM object to stream
        audio from the Focusrite Scarlett 18i20 to the Raspberry Pi4 (raspian) through alsa.
        After the PCM is correctly initialized the loop() is activated for reading data
        """
        #try:
        self.PCM = alsa.PCM(type = alsa.PCM_CAPTURE, mode = alsa.PCM_NORMAL, rate = self.fs, channels = 10, format = alsa.PCM_FORMAT_S32_LE, periodsize = self.blocksize, periods = 10, device = self.device)

        self.PCM.set_tstamp_mode(alsa.PCM_TSTAMP_ENABLE)
        self.PCM.set_tstamp_type(alsa.PCM_TSTAMP_TYPE_GETTIMEOFDAY)

        logger.info(self.PCM.info())

        while self.PCM is not None:
            if not self._state.stop_event.isSet():
                if self.loop():
                    self.stop()
                    return __name__     
            else:
                self.stop()
                logger.info("Alsa Audio source properly stopped after global stop event")
                return None
            
            
        # except KeyboardInterrupt:
        #     logger.info("PCM stopped by user")
        #     self.stop()
        #     return __name__

        # except Exception as e:
        #     logger.error("Error in alsa_input: ", e)
        #     self.stop()
        #     return __name__

    def stop(self):
        """Closes the alsa stream and releases resources."""
        self.PCM.close()
        logger.info("PCM closed")


    def compute_max(self, x):
        mag = []
        thr = 0.01
        m = np.max(x.astype(np.float32), axis=0)
        for i in m:
            if i > thr:
                mag.append('A')
            else: 
                mag.append('_')
        return mag