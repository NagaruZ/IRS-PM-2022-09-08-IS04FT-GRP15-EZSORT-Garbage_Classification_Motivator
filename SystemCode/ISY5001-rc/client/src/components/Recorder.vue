<template>
  <div class="recorder">
      <button class="button" @click="toggleRecorder()">
        <span v-if="!mediaRecorder">Start Recorder</span>
        <span v-else>Stop Recorder</span>
      </button>
      <audio v-if="newAudio" :src="newAudioURL" id="audioPlay" controls></audio>
  </div>
</template>

<script>


export default {
  name: 'RecorderItem',
  data() {
    return {
      mediaRecorder: null,
      stream: null,
      newAudio: null,
      recordedChunks: [],
      audioMIMEString: null,
    }
  },
  computed: {
    newAudioURL() {
      return URL.createObjectURL(this.newAudio)
    }
  },
  methods: {
    startRecording () {
      const secondsOfAudio = 2

      const constraints = (window.constraints = {
        audio: true,
        video: false
      })

      navigator.mediaDevices
        .getUserMedia(constraints)
        .then(stream => {
          this.stream = stream
          this.mediaRecorder = new MediaRecorder(stream)
          this.newAudio = null
          this.recordedChunks = []
          this.mediaRecorder.addEventListener("dataavailable", e => {
            if (e.data.size > 0) {
              this.recordedChunks.push(e.data)
            }
          })
          this.mediaRecorder.start(secondsOfAudio*1000)
        })
        .catch(error => {
          alert(error, "May the browser didn't support or there is some errors.")
        })

      setTimeout(this.stopRecording, secondsOfAudio*1000)
    },
    stopAudioStream () {
      const tracks = this.stream.getTracks()
      tracks.forEach(track => {
        track.stop()
      })
      console.log('Recorder closed')
    },
    async stopRecording() {
      this.mediaRecorder.addEventListener("stop", async () => {
        // when trying to convert the audio to WAV format, simply changing param 'type' does not work
        // this.newAudio = new Blob(this.recordedChunks, {type: 'audio/wav; codecs=MS_PCM'});
        // while, an working way is: Blob --> AudioBuffer --> WAV file
        this.newAudio = new Blob(this.recordedChunks)
        this.newAudioArrayBuffer = await this.newAudio.arrayBuffer()
        let toWav = require('audiobuffer-to-wav')
        let converter = require('base64-arraybuffer')


        const context = new AudioContext();
        await context.decodeAudioData(this.newAudioArrayBuffer, (buffer) => {
          const wavFile = toWav(buffer);
          this.audioMIMEString = converter.encode(wavFile)
          console.log(this.audioMIMEString)
          this.$emit('captured-audio', this.audioMIMEString)
        })
      })
      this.mediaRecorder.stop()
      this.stopAudioStream()
      this.mediaRecorder = null
      this.stream = null
    },
    toggleRecorder() {
      if (!this.mediaRecorder) {
        this.startRecording()
      } else {
        this.stopRecording()
      }
    },
  }
}
</script>


<!--<style>-->

<!--.wrapper {-->
<!--  position: relative;-->
<!--  display: flex;-->
<!--  align-items: center;-->
<!--  justify-content: center;-->
<!--  flex-direction: column;-->
<!--  width: 50%;-->
<!--  height: 90%;-->
<!--  background-color: white;-->
<!--  border: solid 2px rgb(0, 0, 0);-->
<!--}-->

<!--button {-->
<!--  border: solid 1px rgb(0, 0, 0);-->
<!--  /*font-size: 10px;*/-->
<!--  cursor: pointer;-->
<!--}-->

<!--.button {-->
<!--  width: 140px;-->
<!--  height: 40px;-->
<!--}-->
<!--</style>-->