/*! Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: Apache-2.0 */

// All helpers are expose on 'media-devices.js' and 'dom.js'
// const { setupParticipant } = window;
// const { initializeDeviceSelect, getCamera, getMic } = window;

const { Stage, LocalStageStream, SubscribeType, StageEvents, ConnectionState, StreamType } = IVSBroadcastClient;

let cameraButton = document.getElementById("camera-control");
let micButton = document.getElementById("mic-control");
let joinButton = document.getElementById("join-button");
let leaveButton = document.getElementById("leave-button")

let controls = document.getElementById("local-controls");
let videoDevicesList = document.getElementById("video-devices");
let audioDevicesList = document.getElementById("audio-devices");

// Stage management
let stage;
let joining = false;
let connected = false;
let localCamera;
let localMic;
let cameraStageStream;
let micStageStream;
let remoteStreams = [];

const init = async () => {
  await initializeDeviceSelect();
  
  cameraButton.addEventListener('click', () => {
    const isMuted = !cameraStageStream.isMuted
    cameraStageStream.setMuted(isMuted)
    cameraButton.innerText = isMuted ? 'Show Camera' : 'Hide Camera'
  })

  micButton.addEventListener('click', () => {
    const isMuted = !micStageStream.isMuted
    micStageStream.setMuted(isMuted)
    micButton.innerText = isMuted ? 'Unmute Mic' : 'Mute Mic'
  })

  joinButton.addEventListener('click', () => {
    joinStage()
  })
  
  leaveButton.addEventListener('click', () => {
    leaveStage()
  })
};

const joinStage = async () => {
  if (connected || joining) { return }
  joining = true

  const token = document.getElementById("token").value;

  if (!token) {
    window.alert("Please enter a participant token");
    joining = false;
    return;
  }
  
  // Retrieve the User Media currently set on the page
  localCamera = await getCamera(videoDevicesList.value);
  localMic = await getMic(audioDevicesList.value);
  
  // Create StageStreams for Audio and Video
  cameraStageStream = new LocalStageStream(localCamera.getVideoTracks()[0]);
  micStageStream = new LocalStageStream(localMic.getAudioTracks()[0]);
  
  const strategy = {
    stageStreamsToPublish() {   
      return [cameraStageStream, micStageStream];
    },
    shouldPublishParticipant() {
      return true;
    },
    shouldSubscribeToParticipant() {
      return SubscribeType.AUDIO_VIDEO;
    }
  }
  
  stage = new Stage(token, strategy);
  
  // Other available events:
  // https://aws.github.io/amazon-ivs-web-broadcast/docs/sdk-guides/stages#events

  stage.on(StageEvents.STAGE_CONNECTION_STATE_CHANGED, (state) => {
    connected = state === ConnectionState.CONNECTED;

    if (connected) {
      joining = false;
      controls.classList.remove('hidden');
    } else {
      controls.classList.add('hidden')  
    }
  });

  stage.on(StageEvents.STAGE_PARTICIPANT_JOINED, (participant) => {
    console.log("Participant Joined:", participant);
  });

  stage.on(StageEvents.STAGE_PARTICIPANT_STREAMS_ADDED, (participant, streams) => {
    console.log("Participant Media Added: ", participant, streams);

    let streamsToDisplay = streams;

    if (participant.isLocal) {
      // Ensure to exclude local audio streams, otherwise echo will occur
      streamsToDisplay = streams.filter(stream => stream.streamType === StreamType.VIDEO);
    }

    const videoEl = setupParticipant(participant);
    streamsToDisplay.forEach(stream => videoEl.srcObject.addTrack(stream.mediaStreamTrack));
  });
  
  stage.on(StageEvents.STAGE_PARTICIPANT_LEFT, (participant) => {
    console.log("Participant Left: ", participant);
    teardownParticipant(participant);
  });

  try {
    await stage.join();
  } catch (err) {
    joining = false;
    connected = false;
    console.error(err.message);
  }
}

const leaveStage = async () => {
  stage.leave();
  
  joining = false;
  connected = false;
  
  cameraButton.innerText = 'Hide Camera';
  micButton.innerText = 'Mute Mic';
  controls.classList.add('hidden');
}

init();