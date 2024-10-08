"""
Avatar Networking Controller

Usage:
    anc <network> <out_vec_len> [--voice_channels=<spk_chan>]
    anc -h | --help
    anc --version

Options:
    <network>                        The network config to use (adhoc, wifi, localhost)
    <out_vec_len>                    The number of motors in the avatar
    --voice_channels=<spk_chan>   The number of speaker channels. [default: 1]
"""

import asyncio
import threading

import zmq
import zmq.asyncio
from robonet.receive_callbacks import receive_objs
from robonet.buffers.buffer_objects import TensorBuffer
from robonet.buffers.buffer_handling import pack_obj
from robonet.util import send_burst
from docopt import docopt
from typing import Dict, Any
import numpy as np
import random
from displayarray import display
from robonet.buffers.buffer_objects import AudioBuffer
from robonet.receive_callbacks import create_pyramid
from robonet import camera
import cv2
import threading
class AvatarBrain:
    def __init__(self, out_vec_len=8, voice_channels=1, voice_fft_size=1536, camera_resolution=(640, 480), mic_channels=1):
        self.out_vec_len = out_vec_len
        self.voice_channels = voice_channels
        self.camera_resolution = camera_resolution
        self.mic_channels = mic_channels
        self.voice_fft_size = voice_fft_size

        # Data buffers
        self.motor_data = np.zeros([self.out_vec_len], dtype=np.float32)
        self.voice_data = np.zeros([self.voice_fft_size, self.voice_channels], dtype=np.float32)
        self.camera_data = None
        self.mic_data = None

        # Locks for thread-safe access
        self.motor_lock = threading.Lock()
        self.voice_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.mic_lock = threading.Lock()

    # Motor data handling
    def set_motor_data(self, data):
        with self.motor_lock:
            self.motor_data = data

    def get_motor_data(self):
        with self.motor_lock:
            return self.motor_data

    # Voice data handling
    def set_voice_data(self, data):
        with self.voice_lock:
            self.voice_data = data

    def get_voice_data(self):
        with self.voice_lock:
            return self.voice_data

    # Camera data handling
    def set_camera_data(self, data):
        with self.camera_lock:
            self.camera_data = data

    def get_camera_data(self):
        with self.camera_lock:
            return self.camera_data

    # Mic data handling
    def set_mic_data(self, data):
        with self.mic_lock:
            self.mic_data = data

    def get_mic_data(self):
        with self.mic_lock:
            return self.mic_data

    async def run(self):
        while True:
            self.set_motor_data(np.random.random([self.out_vec_len]).astype(np.float32))
            self.set_voice_data(np.random.random([self.voice_fft_size, self.voice_channels]).astype(np.float32))

            print('hi')
            await asyncio.sleep(0)
async def transmit_motor_random(controller, radio_lock, radio):
    """Async function to transmit random motor data"""
    print("transmitting random motor data..")
    while True:
        #out_arr = np.random.random([controller.out_vec_len])
        out_arr = controller.get_motor_data()  # gets latest motor data, non-blocking

        # Send direct messages to the server
        direct_message = TensorBuffer([out_arr])
        direct_message = pack_obj(direct_message)
        parts = [direct_message[i:i + 4096] for i in range(0, len(direct_message), 4096)]

        send_burst(radio_lock, radio, bytes([random.randint(0, 255)]), parts)
        print(f"Sent motor frame")
        await asyncio.sleep(0)

async def transmit_voice_random(controller, radio_lock, radio):
    """Async function to transmit random voice data"""
    print("transmitting random voice data..")
    while True:
        #out_arr = np.random.random([fft_size, controller.voice_channels])
        out_arr = controller.get_voice_data()  # gets latest voice data, non-blocking

        # Send direct messages to the server
        # todo: this needs to be distinct from the motor tensor. For now, maybe send audio data. Later, label buffers.
        direct_message = TensorBuffer([out_arr])
        direct_message = pack_obj(direct_message)
        parts = [direct_message[i:i + 4096] for i in range(0, len(direct_message), 4096)]

        send_burst(radio_lock, radio, bytes([random.randint(0, 255)]), parts)
        print(f"Sent voice frame")
        await asyncio.sleep(0)


def display_fftnet(displayer, controller):
    def fft_to_nnet(obj:AudioBuffer):
        fft_size = obj.fft_data.shape[0]*2
        fft_mag = np.abs(obj.fft_data) / (fft_size // 2)

        # magnify lower amplitudes
        fft_mag = np.sqrt(fft_mag)  # sounddevice sets mag to -1 to 1, so sqrt is fine
        # this should go through an edge detector, just like vision

        fft_phase = np.angle(obj.fft_data)  # -pi to pi phase
        # this should go through a convolution, but maybe not an edge detector, so it needs to learn

        fft_parts = obj.fft_data[..., np.newaxis].view(np.float32)  # -1 to 1 complex plane
        # also through conv, learned
        # sqrt(a^2+b^2), atan2(a, b), and back aren't easy functions for a neural net to learn, so this is useful info

        fft_list = []
        for i in range(fft_mag.shape[1]):
            fft_list.append(fft_mag[:, i])
            fft_list.append(fft_phase[:, i])
            fft_list.append(fft_parts[:, i, 0])
            fft_list.append(fft_parts[:, i, 1])

        full_fft = np.stack(fft_list, axis=-1)

        fft_pyr = create_pyramid(full_fft)
        controller.set_mic_data(fft_pyr)

        for e, fft_p in enumerate(fft_pyr):
            displayer.update(fft_p, f'fft {e}')

    return fft_to_nnet

def display_mjpg_cv(displayer, controller):
    def display_mjpeg(obj):
        imgb = camera.CameraPack.unpack_frame(obj.mjpeg)
        img = camera.CameraPack.to_cv2_image(imgb)
        try:
            if img is not None and img.size > 0:
                controller.set_camera_data(img)
                displayer.update(img, 'Camera Stream')
        except cv2.error as e:
            print(f"OpenCV error: {e}")


    return display_mjpeg

async def udp_loop(ctx, local_ip, client_ip, args):
    out_vec_len = int(args['<out_vec_len>'])
    voice_channels = int(args['--voice_channels'])

    radio_lock = threading.Lock()

    unicast_radio = ctx.socket(zmq.RADIO)
    unicast_radio.setsockopt(zmq.LINGER, 0)
    unicast_radio.setsockopt(zmq.CONFLATE, 1)
    unicast_dish = ctx.socket(zmq.DISH)
    unicast_dish.setsockopt(zmq.LINGER, 0)
    unicast_dish.setsockopt(zmq.CONFLATE, 1)
    unicast_dish.rcvtimeo = 1000  # check messages quickly and then move to other IO threads

    unicast_dish.bind(f"udp://{local_ip}:9998")
    unicast_dish.join("direct")
    unicast_radio.connect(f"udp://{client_ip}:9999")

    controller = AvatarBrain(out_vec_len=out_vec_len, voice_channels=voice_channels)
    d = display()

    await asyncio.gather(
        controller.run(), # todo: this is compute heavy. Run it in another thread.
        transmit_motor_random(controller, radio_lock, unicast_radio),
        transmit_voice_random(controller, radio_lock, unicast_radio),
        receive_objs({'AudioBuffer': display_fftnet(d, controller), 'MJpegCamFrame': display_mjpg_cv(d, controller)})(unicast_radio, unicast_dish)
    )

    unicast_radio.close()
    unicast_dish.close()


async def main(args: Dict[str, Any]):
    ctx = zmq.Context.instance()

    network = args['<network>']

    if network == 'localhost':
        local_ip = other_ip = '127.0.0.1'
        await udp_loop(ctx, local_ip, other_ip, args)
    elif network in ['wifi', 'adhoc']:
        from robonet.util import get_local_ip, server_udp_discovery
        local_ip = get_local_ip()
        other_ip = server_udp_discovery(ctx, local_ip)
        if network == 'adhoc':
            from robonet.buffers.buffer_objects import WifiSetupInfo
            from robonet.util import get_connection_info, switch_connections
            from robonet.adhoc_pair.server import lazy_pirate_send_con_info, set_hotspot
            wifi_obj = WifiSetupInfo("robot_wifi", "192.168.2.1", "192.168.2.2")
            lazy_pirate_send_con_info(ctx, wifi_obj, local_ip)

            devices, current_connection = get_connection_info()
            try:
                set_hotspot(wifi_obj, devices)
                await udp_loop(ctx, wifi_obj.server_ip, wifi_obj.client_ip, args)

            finally:
                switch_connections(wifi_obj.ssid, current_connection)
                ctx.term()
        else:
            await udp_loop(ctx, local_ip, other_ip, args)
    else:
        raise ValueError(f"Unknown Network Type: {network}")


if __name__ == "__main__":
    args = docopt(__doc__, version='Avatar Networking Controller 0.1')

    asyncio.run(main(args))
