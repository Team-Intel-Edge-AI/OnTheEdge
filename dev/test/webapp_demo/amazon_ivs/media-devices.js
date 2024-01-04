/*! Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: Apache-2.0 */

/**
 * Returns an initial list of devices populated on the page selects
 */
async function initializeDeviceSelect() {
    const videoSelectEl = document.getElementById('video-devices');
    videoSelectEl.disabled = false;

    const { videoDevices, audioDevices } = await getDevices();
    videoDevices.forEach((device, index) => {
        videoSelectEl.options[index] = new Option(device.label, device.deviceId);
    });

    const audioSelectEl = document.getElementById('audio-devices');

    audioSelectEl.disabled = false;
    audioDevices.forEach((device, index) => {
        audioSelectEl.options[index] = new Option(device.label, device.deviceId);
    });
}

/**
 * Returns all devices available on the current device
 */
async function getDevices() {
    // Prevents issues on Safari/FF so devices are not blank
    await navigator.mediaDevices.getUserMedia({ video: true, audio: true });

    const devices = await navigator.mediaDevices.enumerateDevices();
  
    // Get all video devices
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');
    if (!videoDevices.length) {
        console.error('No video devices found.');
    }
  
    // Get all audio devices
    const audioDevices = devices.filter((d) => d.kind === 'audioinput');
    if (!audioDevices.length) {
        console.error('No audio devices found.');
    }

    return { videoDevices, audioDevices };
}

async function getCamera(deviceId) {
    console.log(deviceId)
    // Use Max Width and Height
    return navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : null, 
        },
        audio: false,
    });
}

async function getMic(deviceId) {
    console.log(deviceId)
    return navigator.mediaDevices.getUserMedia({
        video: false,
        audio: {
          deviceId: deviceId ? { exact: deviceId } : null,
        },
    });
}
