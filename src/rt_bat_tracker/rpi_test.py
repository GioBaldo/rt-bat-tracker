import sounddevice as sd
print("PortAudio version:", sd.get_portaudio_version())
print("\nDevices:")
for i, d in enumerate(sd.query_devices()):
    print(i, d)

print("\nHost APIs:")
for i, h in enumerate(sd.query_hostapis()):
    print(i, h)

print("\nDefault device:", sd.default.device)


DEV = 1   # metti l'indice della Scarlett hw:3,0
rates = [44100, 48000, 96000, 192000]
chs   = [1, 2, 10, 18, 26]
dtypes = ['float32', 'int32', 'int16', 'pippo']

for rate in rates:
    for ch in chs:
        for dt in dtypes:
            try:
                sd.check_input_settings(device=DEV, samplerate=rate, channels=ch, dtype=dt)
                print("OK ", rate, ch, dt)
            except Exception as e:
                print("ERR", rate, ch, dt, "->", e)