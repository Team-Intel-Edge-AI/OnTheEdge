const express = require("express");
const app = express();
const fs = require("fs");


// const request = require('request');
// http.createServer(function (req, resp) {
//   if (req.url === '/doodle.png') {
//     if (req.method === 'GET') {
//       request.get('http://example.com/doodle.png').pipe(resp)
//     }
//   }
// })

// app.init(joinStage());

app.get("/", function (req, res) {
    res.sendFile(__dirname + "/index.html");
});

app.get("/video", function (req, res) {
    const range = req.headers.range;
    if (!range) {
        res.status(400).send("Requires Range header");
    }

    const videoPath = "Chris-Do.mp4";
    const videoSize = fs.statSync("Chris-Do.mp4").size;

    const CHUNK_SIZE = 10 ** 6; // 1MB
    const start = Number(range.replace(/\D/g, ""));
    const end = Math.min(start + CHUNK_SIZE, videoSize - 1);

    // Create headers
    const contentLength = end - start + 1;
    const headers = {
        "Content-Range": `bytes ${start}-${end}/${videoSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": contentLength,
        "Content-Type": "video/mp4",
    };

    // HTTP Status 206 for Partial Content
    res.writeHead(206, headers);

    // create video read stream for this particular chunk
    const videoStream = fs.createReadStream(videoPath, { start, end });

    // Stream the video chunk to the client
    videoStream.pipe(res);
});

app.listen(8000, function () {
    console.log("Listening on port 8000!");
});



const { Stage, LocalStageStream, RemoteStageStream, SubscribeType, StageEvents, ConnectionState, StreamType } = IVSBroadcastClient;

// Stage management
let stage;
let joining = false;
let connected = false;
let cameraStageStream;
let micStageStream;


const joinStage = async () => {
    if (connected || joining) { return }
    joining = true
  
    const token = "eyJhbGciOiJLTVMiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3MDU1NTc0MDksImlhdCI6MTcwNDM0NzgwOSwianRpIjoiS2hTRmlibnZCUVNCIiwicmVzb3VyY2UiOiJhcm46YXdzOml2czphcC1ub3J0aGVhc3QtMjo5MzUwOTYzMTI5NDM6c3RhZ2UvOGg4NnlXdXZuZjE3IiwidG9waWMiOiI4aDg2eVd1dm5mMTciLCJldmVudHNfdXJsIjoid3NzOi8vZ2xvYmFsLmVldmVlLmV2ZW50cy5saXZlLXZpZGVvLm5ldCIsIndoaXBfdXJsIjoiaHR0cHM6Ly82NWFjYTJmZmYwYTIuZ2xvYmFsLWJtLndoaXAubGl2ZS12aWRlby5uZXQiLCJ1c2VyX2lkIjoici1zZXJ2ZXIiLCJjYXBhYmlsaXRpZXMiOnsiYWxsb3dfcHVibGlzaCI6dHJ1ZSwiYWxsb3dfc3Vic2NyaWJlIjp0cnVlfSwidmVyc2lvbiI6IjAuMCJ9.MGUCMQCTatOcVDK1JVojebCIeasE36rttY-k7gBqZ_uzd3uTg0sYGHbcsmZIjLsLy3tmouECMBBlA1OGvrA4QNrzBv7RNhnV4mQiadmX32Gv5cBbWrSSYLD_XESev8CK9JWf3YqTOQ";
    
    // // Create StageStreams for Audio and Video
    // cameraStageStream = new LocalStageStream();
    // micStageStream = new LocalStageStream();
    
    const strategy = {
        stageStreamsToPublish() {   
            // return [cameraStageStream, micStageStream];
            return [];
        },
        shouldPublishParticipant() {
            return false;
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

      streamsToDisplay.forEach(stream => print(stream.mediaStreamTrack));
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

joinStage();
