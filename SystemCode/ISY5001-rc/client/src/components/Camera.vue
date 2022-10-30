<template>
  <div class="camera">
       <button class="button" @click="toggleCamera()">
        <span v-if="!isCameraOpen">Open Camera</span>
        <span v-else>Close Camera</span>
      </button>
      <div class="video-container">
        <video class="camera-video" v-show="isCameraOpen && !isPhotoTaken" ref="camera" :width="450" :height="337" autoplay playsinline ></video>
        <canvas id="photoTaken" v-show="isPhotoTaken" class="canvas-photo" ref="canvas" :width="450" :height="337"></canvas>
      </div>
      <button v-if="isCameraOpen && !isPhotoTaken" class="button" @click="takePhoto()">
        <span>Snap!</span>
      </button>
      <button v-if="!isCameraOpen && isPhotoTaken" class="button" @click="openCamera()">
        <span>Take another photo</span>
      </button>
      <button v-if="isPhotoTaken" class="camera-download">
        <a id="downloadPhoto" download="Capture.jpg" class="button" role="button" @click="downloadImage">
          Download
        </a>
      </button>

  </div>
</template>

<script>
export default {
  name: 'CameraItem',
  data() {
    return {
      isCameraOpen: false,
      isPhotoTaken: false,
      photoFromVideo: null,
      photoMIMEString: null,
    }
  },
  methods: {
    createCameraElement () {
      const constraints = (window.constraints = {
        audio: false,
        video: true
      })

      navigator.mediaDevices
        .getUserMedia(constraints)
        .then(stream => {
          this.$refs.camera.srcObject = stream
        })
        .catch(error => {
          alert(error, "May the browser didn't support or there is some errors.")
        })
    },
    stopCameraStream () {
      const tracks = this.$refs.camera.srcObject.getTracks()
      tracks.forEach(track => {
        track.stop()
      })
      console.log('Camera closed')
    },
    closeCamera() {
      this.stopCameraStream()
      this.isCameraOpen = false
    },
    openCamera() {
      this.isCameraOpen = true
      this.isPhotoTaken = false // reset
      this.createCameraElement()
    },
    toggleCamera () {
      if (this.isCameraOpen) {
        this.closeCamera()

        this.isPhotoTaken = false
        console.log('Camera closed');
      } else {
        this.openCamera()
        console.log('Camera open');
      }
    },
    takePhoto () {
      this.isPhotoTaken = !this.isPhotoTaken

      const context = this.$refs.canvas.getContext('2d')
      this.photoFromVideo = this.$refs.camera
      context.drawImage(this.photoFromVideo, 0, 0, 450, 337)

      this.photoMIMEString = document.getElementById("photoTaken").toDataURL("image/jpeg")
        .replace("data:image/jpeg;base64,", ""); // remove MIME header
      this.$emit('captured-image', this.photoMIMEString)

      this.closeCamera() // close camera after taking a photo
    },
    downloadImage() {
      const download = document.getElementById("downloadPhoto");
      const canvas = document.getElementById("photoTaken").toDataURL("image/jpeg")
        .replace("image/jpeg", "image/octet-stream")
      download.setAttribute("href", canvas);
    }
  }
}
</script>

<!--<style>-->
<!--.camera {-->
<!--  position: fixed;-->
<!--  top: 0;-->
<!--  left: 0;-->
<!--  width: 100vw;-->
<!--  height: 100vh;-->
<!--  overflow: hidden;-->
<!--  background: rgba(0, 0, 0, 0.45);-->
<!--  display: flex;-->
<!--  align-items: center;-->
<!--  justify-content: center;-->
<!--}-->

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

<!--.video-container {-->
<!--  display: flex;-->
<!--  flex-direction: column;-->
<!--}-->
<!--</style>-->