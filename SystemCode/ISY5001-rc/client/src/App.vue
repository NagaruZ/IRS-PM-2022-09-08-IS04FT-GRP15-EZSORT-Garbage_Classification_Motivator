<template>
  <div class="app">
      <div class="container text-center">
        <div class="row">
          <div class="col">
            <div class="card">
              <div class="card-body">
                <recorder-item @captured-audio="handleCapturedAudio"></recorder-item>
              </div>
            </div>
          </div>
          <div class="col" v-show="capturedAudio">
            <div class="card">
              <div class="card-body">
                <detection-result-item :capturedAudio="capturedAudio"></detection-result-item>
              </div>
            </div>
          </div>
        </div>
      </div>
  </div>

</template>

<script>
  // import CameraItem from './components/Camera.vue'
  import RecorderItem from './components/Recorder'
  import DetectionResultItem from "@/components/DetectionResult";

  export default {
    name: 'App',
    data() {
      return {
        capturedImage: null,
        capturedAudio: null,
      }
    },
    components: {
        DetectionResultItem,
        // CameraItem,
        RecorderItem,
    },
    methods: {
      handleCapturedImage(data) {
        this.capturedImage = data
      },
      handleCapturedAudio(data) {
        this.capturedAudio = data
      }
    },
  }
</script>

<style>
  * {
    margin: 0;
    padding: 0;
  }
  .app {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'VT323', monospace;
  }
  .main {
    width: 30%;
    height: 35%;
    border: solid 1px rgb(0, 0, 0);
    border-radius: 5px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  h1 {
    font-size: 50px;
  }
  p {
    font-size: 30px;
    margin-bottom: 20px;
  }
  .camera-icon {
    width: 25px;
    height: 25px;
  }
  .camera-icon:hover {
    cursor: pointer;
  }
</style>