const { Stage, LocalStageStream, SubscribeType, StageEvents, ConnectionState, StreamType } = IVSBroadcastClient;

let joinButton = document.getElementById("join-button");
let leaveButton = document.getElementById("leave-button");

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

  const token = "eyJhbGciOiJLTVMiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3MDU1NTYyODksImlhdCI6MTcwNDM0NjY4OSwianRpIjoic2tReTJQVGZEZDE0IiwicmVzb3VyY2UiOiJhcm46YXdzOml2czphcC1ub3J0aGVhc3QtMjo5MzUwOTYzMTI5NDM6c3RhZ2UvOGg4NnlXdXZuZjE3IiwidG9waWMiOiI4aDg2eVd1dm5mMTciLCJldmVudHNfdXJsIjoid3NzOi8vZ2xvYmFsLmVldmVlLmV2ZW50cy5saXZlLXZpZGVvLm5ldCIsIndoaXBfdXJsIjoiaHR0cHM6Ly82NWFjYTJmZmYwYTIuZ2xvYmFsLWJtLndoaXAubGl2ZS12aWRlby5uZXQiLCJ1c2VyX2lkIjoiY2FtZXJhLWNsaWVudCIsImNhcGFiaWxpdGllcyI6eyJhbGxvd19wdWJsaXNoIjp0cnVlLCJhbGxvd19zdWJzY3JpYmUiOnRydWV9LCJ2ZXJzaW9uIjoiMC4wIn0.MGUCMGxjp_ffWxOdiEsAjAHb84EwqtL-OTu4-9huWg5LumH5ZIv6YMBE3Oow9zybJXpJaAIxAOAFnT7JA_tpPz37f36-GroLb7GlI3gE-RnAI-gQ1CbQTSRm17Nhq7iwdqT2Ro4icQ";
  
  // Retrieve the User Media currently set on the page
  localCamera = await getCamera('');
  localMic = await getMic('');
  
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
  
  stage.on(StageEvents.STAGE_CONNECTION_STATE_CHANGED, (state) => {
    connected = state === ConnectionState.CONNECTED;
    if (connected) {
        joining = false;
        console.log('Connected');
    } else {
        console.log('Not connected');
    }
  });

  stage.on(StageEvents.STAGE_PARTICIPANT_JOINED, (participant) => {
    console.log("Participant Joined:", participant);
  });

  stage.on(StageEvents.STAGE_PARTICIPANT_STREAMS_ADDED, (participant, streams) => {
    console.log("Participant Media Added: ", participant, streams);
  });
  
  stage.on(StageEvents.STAGE_PARTICIPANT_LEFT, (participant) => {
    console.log("Participant Left: ", participant);
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

async function getCamera(deviceId) {
    // Use Max Width and Height
    return navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : null, 
        },
        audio: false,
    });
}

async function getMic(deviceId) {
    return navigator.mediaDevices.getUserMedia({
        video: false,
        audio: {
          deviceId: deviceId ? { exact: deviceId } : null,
        },
    });
}

init();