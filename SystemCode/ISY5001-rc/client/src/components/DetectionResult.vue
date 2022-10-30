<template>
  <div class="container">
    <button v-if="!isRequestSent && capturedAudio" id="detectAudio" @click="detectAudio()">Detect audio!</button>
    <div v-else id="detection-result">
      <div id="body">{{ msg.message }}</div>
      <button @click="newDetection()">New detection</button>
    </div>
    <div v-show="false">{{ capturedAudio }}</div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'DetectionResultItem',
  data() {
    return {
      msg: '',
      isRequestSent: false
    };
  },
  props: ['capturedAudio'],
  methods: {
    detectAudio() {
      this.isRequestSent = true
      let param = {'audio': this.capturedAudio}
      const path = '/detectAudio';
      axios.post(path, param)
          .then((res) => {
            console.log(res)
            this.msg = res.data;
            window.location.replace("http://127.0.0.1:5000/showResult?label=" + this.msg.message)
          })
          .catch((error) => {
            // eslint-disable-next-line
            console.error(error);
          });
    },
    newDetection() {
      this.isRequestSent = false
    }
  },
};
</script>