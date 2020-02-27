import streamlit as st
import numpy as np
import scipy.io.wavfile as wf
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd

# Live recording
import sounddevice as sd
import soundfile as sf

def add_activity():
	st.title("Voice Activity Detection")

	st.write("This application demonstrates a simple Voice Activity Detection algorithm that works for any language.")
	
	st.sidebar.title("Parameters")
	duration = st.sidebar.slider("Recording duration", 0.0, 10.0, 3.0)
	threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.6)
	sample_window = st.sidebar.slider("Window size", 0.0, 0.15, 0.02)
	speech_window = st.sidebar.slider("Speech Window", 0.0, 1.0, 0.5)  

	if st.button("Start Recording"):
	    with st.spinner("Recording..."):
	    	sr = 16000
	    	audio_bytes = sd.rec(int(duration * sr), samplerate=sr, channels=1).reshape(-1)
	    	sd.wait()

	class VoiceActivityDetector():
	    """ Use signal energy to detect voice activity in wav file """
	    
	    def __init__(self, wave_input_filename, speech_energy_threshold, sample_window, speech_window):
	        self._read_wav(wave_input_filename)._convert_to_mono()
	        self.sample_window = sample_window #20 ms
	        self.sample_overlap = 0.01 #10ms
	        self.speech_window = speech_window #half a second
	        self.speech_energy_threshold = speech_energy_threshold #60% of energy in voice band
	        self.speech_start_band = 300
	        self.speech_end_band = 3000
	           
	    def _read_wav(self, wave_file):
	        self.rate, self.data = wf.read(wave_file)
	        self.channels = len(self.data.shape)
	        self.filename = wave_file
	        return self
	    
	    def _convert_to_mono(self):
	        if self.channels == 2 :
	            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
	            self.channels = 1
	        return self
	    
	    def _calculate_frequencies(self, audio_data):
	        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)
	        data_freq = data_freq[1:]
	        return data_freq    
	    
	    def _calculate_amplitude(self, audio_data):
	        data_ampl = np.abs(np.fft.fft(audio_data))
	        data_ampl = data_ampl[1:]
	        return data_ampl
	        
	    def _calculate_energy(self, data):
	        data_amplitude = self._calculate_amplitude(data)
	        data_energy = data_amplitude ** 2
	        return data_energy
	        
	    def _znormalize_energy(self, data_energy):
	        energy_mean = np.mean(data_energy)
	        energy_std = np.std(data_energy)
	        energy_znorm = (data_energy - energy_mean) / energy_std
	        return energy_znorm
	    
	    def _connect_energy_with_frequencies(self, data_freq, data_energy):
	        energy_freq = {}
	        for (i, freq) in enumerate(data_freq):
	            if abs(freq) not in energy_freq:
	                energy_freq[abs(freq)] = data_energy[i] * 2
	        return energy_freq
	    
	    def _calculate_normalized_energy(self, data):
	        data_freq = self._calculate_frequencies(data)
	        data_energy = self._calculate_energy(data)
	        #data_energy = self._znormalize_energy(data_energy) #znorm brings worse results
	        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
	        return energy_freq
	    
	    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):
	        sum_energy = 0
	        for f in energy_frequencies.keys():
	            if start_band<f<end_band:
	                sum_energy += energy_frequencies[f]
	        return sum_energy
	    
	    def _median_filter (self, x, k):
	        assert k % 2 == 1, "Median filter length must be odd."
	        assert x.ndim == 1, "Input must be one-dimensional."
	        k2 = (k - 1) // 2
	        y = np.zeros ((len (x), k), dtype=x.dtype)
	        y[:,k2] = x
	        for i in range (k2):
	            j = k2 - i
	            y[j:,i] = x[:-j]
	            y[:j,i] = x[0]
	            y[:-j,-(i+1)] = x[j:]
	            y[-j:,-(i+1)] = x[-1]
	        return np.median (y, axis=1)
	        
	    def _smooth_speech_detection(self, detected_windows):
	        median_window=int(self.speech_window/self.sample_window)
	        if median_window%2==0: median_window=median_window-1
	        median_energy = self._median_filter(detected_windows[:,1], median_window)
	        return median_energy
	      
	    def plot_detected_speech_regions(self):
	        """ Performs speech detection and plot original signal and speech regions.
	        """
	        data = self.data
	        detected_windows = self.detect_speech()
	        data_speech = np.zeros(len(data))
	        it = np.nditer(detected_windows[:,0], flags=['f_index'])
	        while not it.finished:
	            data_speech[int(it[0])] = data[int(it[0])] * detected_windows[it.index,1]
	            it.iternext()
	        plt.figure()
	        plt.plot(data_speech)
	        plt.plot(data)
	        plt.show()
	        return self
	       
	    def detect_speech(self):
	        """ Detects speech regions based on ratio between speech band energy
	        and total energy.
	        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).
	        """
	        detected_windows = np.array([])
	        sample_window = int(self.rate * self.sample_window)
	        sample_overlap = int(self.rate * self.sample_overlap)
	        data = self.data
	        sample_start = 0
	        start_band = self.speech_start_band
	        end_band = self.speech_end_band
	        while (sample_start < (len(data) - sample_window)):
	            sample_end = sample_start + sample_window
	            if sample_end>=len(data): sample_end = len(data)-1
	            data_window = data[sample_start:sample_end]
	            energy_freq = self._calculate_normalized_energy(data_window)
	            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
	            sum_full_energy = sum(energy_freq.values())
	            speech_ratio = sum_voice_energy/sum_full_energy
	            # Hypothesis is that when there is a speech sequence we have ratio of energies more than Threshold
	            speech_ratio = speech_ratio > self.speech_energy_threshold
	            detected_windows = np.append(detected_windows,[sample_start, speech_ratio])
	            sample_start += sample_overlap
	        detected_windows = detected_windows.reshape(int(len(detected_windows)/2),2)
	        detected_windows[:,1] = self._smooth_speech_detection(detected_windows)
	        return detected_windows

	v = VoiceActivityDetector(filename, threshold, sample_window, speech_window)
	df = pd.DataFrame(v.data).reset_index()
	df.columns=['Time', 'Frequence']
	c = alt.Chart(df).mark_circle().encode(x='Time', y='Frequence')
	df2 = pd.DataFrame(v.detect_speech(), columns=['Time', 'Frequence'])
	c2 = alt.Chart(df2).mark_line().encode(x='Time', y='Frequence')

	st.sidebar.title("Details")
	st.sidebar.markdown("Based on : https://github.com/marsbroshok/VAD-python")

	st.sidebar.markdown("Input audio data treated as following:")
	st.sidebar.markdown("- Convert stereo to mono") 
	st.sidebar.markdown("- Move a window of 20ms along the audio data") 
	st.sidebar.markdown("- Calculate the ratio between energy of speech band and total energy for window")
	st.sidebar.markdown("- If ratio is more than threshold (0.6 by default) label windows as speech")
	st.sidebar.markdown("- Apply median filter with length of 0.5s to smooth detected speech regions")
	st.sidebar.markdown("- Represent speech regions as intervals of time")

	st.markdown("*Plot the raw audio signal:*")
	st.write(c)

	st.markdown("*Plot the detected voice:*")
	st.write(c2)