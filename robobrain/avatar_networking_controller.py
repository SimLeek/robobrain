"""
Avatar Networking Controller

Usage:
    anc <network> <num_motors> [--speaker_channels=<spk_chan>] [--mic_channels=<mic_chan>] [--camera_resolution=<res>]
    ani -h | --help
    ani --version

Options:
    <network>                        The network config to use (adhoc, wifi, localhost)
    <num_motors>                    The number of motors in the avatar
    --voice_channels=<spk_chan>   The number of speaker channels. [default: 1]
"""

import asyncio
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
from robonet.receive_callbacks import display_fftnet, display_mjpg_cv

async def transmit_motor_random(radio_lock, radio, num_motors=8):
    """Async function to transmit camera MJPEG data"""
    print("transmitting random motor data..")
    while True:
        out_arr = np.random.random([num_motors])
        # Send direct messages to the server
        direct_message = TensorBuffer([out_arr])
        direct_message = pack_obj(direct_message)
        parts = []
        for i in range(0, len(direct_message), 4096):
            parts.append(direct_message[i:i + 4096])

        await send_burst(radio_lock, radio, bytes([random.randint(0, 255)]), parts)

        print(f"Sent frame")
        await asyncio.sleep(0)

async def transmit_voice_random(radio_lock, radio, channels=1, fft_size=1536):
    """Async function to transmit camera MJPEG data"""
    print("transmitting random motor data..")
    while True:
        out_arr = np.random.random([fft_size, channels])
        # Send direct messages to the server
        direct_message = TensorBuffer([out_arr])
        direct_message = pack_obj(direct_message)
        parts = []
        for i in range(0, len(direct_message), 4096):
            parts.append(direct_message[i:i + 4096])

        await send_burst(radio_lock, radio, bytes([random.randint(0, 255)]), parts)

        print(f"Sent frame")
        await asyncio.sleep(0)




async def udp_loop(ctx, local_ip, client_ip, args):
    num_motors = args['<num_motors>']
    voice_channels = int(args['--voice_channels'])

    radio_lock = asyncio.Lock()

    unicast_radio = ctx.socket(zmq.RADIO)
    unicast_radio.setsockopt(zmq.LINGER, 0)
    unicast_radio.setsockopt(zmq.CONFLATE, 1)
    unicast_dish = ctx.socket(zmq.DISH)
    unicast_dish.setsockopt(zmq.LINGER, 0)
    unicast_dish.setsockopt(zmq.CONFLATE, 1)
    unicast_dish.rcvtimeo = 1000

    unicast_dish.bind(f"udp://{local_ip}:9998")
    unicast_dish.join("direct")
    unicast_radio.connect(f"udp://{client_ip}:9999")

    d = display()

    # todo: these should use asyncio locks to gather data from one async ML thread
    await asyncio.gather(
        transmit_motor_random(radio_lock, unicast_radio, num_motors=num_motors),
        transmit_voice_random(radio_lock, unicast_radio, channels=voice_channels),
        receive_objs({'AudioBuffer': display_fftnet(d), 'MJpegCamFrame': display_mjpg_cv(d)})(unicast_dish)
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
