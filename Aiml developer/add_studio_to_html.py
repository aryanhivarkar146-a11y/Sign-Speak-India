import os

index_html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
with open(index_html_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add recording overlay inside the video frame container
overlay_html = """          <!-- Recording Countdown & Status Overlay -->
          <div id="recordingOverlay" class="absolute inset-0 bg-black/40 backdrop-blur-xs flex flex-col items-center justify-center pointer-events-none hidden z-20">
            <div id="recordingCountdownText" class="text-6xl font-black text-white drop-shadow-lg">3</div>
            <p id="recordingSubtext" class="text-white text-sm font-semibold tracking-wide uppercase mt-2">Get Ready to Sign...</p>
            <div class="w-48 bg-gray-700/80 h-2 rounded-full overflow-hidden mt-3">
              <div id="recordingProgressBar" class="bg-red-500 h-full w-0 transition-all duration-75"></div>
            </div>
          </div>"""

if "id=\"recordingOverlay\"" not in html:
    html = html.replace("<canvas id=\"landmarkCanvas\" class=\"absolute inset-0 w-full h-full pointer-events-none\" aria-hidden=\"true\"></canvas>",
                        "<canvas id=\"landmarkCanvas\" class=\"absolute inset-0 w-full h-full pointer-events-none\" aria-hidden=\"true\"></canvas>\n" + overlay_html)
    print("Added recordingOverlay to video container!")

# 2. Add Studio Card below main
studio_html = """
  <!-- AI/ML Data Collection Studio: Browser Landmark Recorder -->
  <section class="max-w-7xl mx-auto mt-6 bg-gradient-to-r from-slate-900 to-indigo-950 text-white rounded-2xl p-5 shadow-lg border border-indigo-900/50">
    <div class="flex flex-wrap items-center justify-between gap-4 mb-4 pb-3 border-b border-indigo-800/60">
      <div class="flex items-center gap-3">
        <span class="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-xl font-bold border border-indigo-500/30">
          🎙️
        </span>
        <div>
          <div class="flex items-center gap-2">
            <h2 class="text-base font-bold tracking-tight text-white">Browser Sample Recording Studio</h2>
            <span class="bg-indigo-500/20 text-indigo-300 text-xs px-2.5 py-0.5 rounded-full border border-indigo-500/30 font-medium">AIML Developer</span>
          </div>
          <p class="text-xs text-indigo-200/70">Capture real webcam 30-frame landmark sequences to eliminate domain shift & calibrate unknown/neutral gestures</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <span id="sessionCounterBadge" class="bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span id="sessionSamplesCount">0</span> Samples Recorded
        </span>
      </div>
    </div>

    <!-- Recorder Controls Grid -->
    <div class="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
      <!-- Target Class Dropdown -->
      <div class="md:col-span-4">
        <label class="block text-xs font-semibold text-indigo-200 mb-1.5">Target Gesture Label</label>
        <div class="relative">
          <select id="recordLabelSelect" onchange="handleLabelSelectChange()" class="w-full bg-slate-800 border border-indigo-700/60 text-white text-sm rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-400 font-medium">
            <optgroup label="Essential Calibration">
              <option value="unknown" selected>⚠️ unknown / no_sign (Resting / Noise / Idle)</option>
            </optgroup>
            <optgroup label="Current 13 Target Classes">
              <option value="actor">actor</option>
              <option value="adult">adult</option>
              <option value="afternoon">afternoon</option>
              <option value="alive">alive</option>
              <option value="alright">alright</option>
              <option value="animal">animal</option>
              <option value="bad">bad</option>
              <option value="bag">bag</option>
              <option value="ball">ball</option>
              <option value="bank">bank</option>
              <option value="bathroom">bathroom</option>
              <option value="beautiful">beautiful</option>
              <option value="doctor">doctor (Need Doctor)</option>
            </optgroup>
            <optgroup label="Emergency & Useful ISL Additions">
              <option value="thankyou">thankyou (Thank You)</option>
              <option value="hospital">hospital</option>
              <option value="medicine">medicine</option>
              <option value="police">police</option>
              <option value="sick">sick</option>
            </optgroup>
            <optgroup label="Custom">
              <option value="__custom__">➕ Type custom sign...</option>
            </optgroup>
          </select>
          <input type="text" id="customLabelInput" placeholder="Enter sign name..." class="hidden w-full mt-2 bg-slate-800 border border-indigo-600 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400 font-medium">
        </div>
      </div>

      <!-- Signer ID Input -->
      <div class="md:col-span-3">
        <label class="block text-xs font-semibold text-indigo-200 mb-1.5">Signer Name / ID</label>
        <input type="text" id="recordSignerInput" value="signer_aryan" placeholder="e.g. signer_1" class="w-full bg-slate-800 border border-indigo-700/60 text-white text-sm rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-400 font-medium">
      </div>

      <!-- Action Buttons -->
      <div class="md:col-span-5 flex flex-wrap gap-2">
        <button id="recordSampleBtn" onclick="startSampleRecording()" class="flex-1 bg-red-600 hover:bg-red-700 active:bg-red-800 text-white font-semibold py-2.5 px-4 rounded-xl transition shadow flex items-center justify-center gap-2 text-sm">
          <span class="w-2.5 h-2.5 rounded-full bg-white animate-pulse"></span>
          Record Sample (30 Frames)
        </button>
        <button onclick="recordBurstIdleSamples()" class="bg-indigo-700/70 hover:bg-indigo-600 text-white font-medium py-2.5 px-3.5 rounded-xl transition text-xs flex items-center gap-1.5 border border-indigo-500/40">
          ⚡ Quick 3x Idle
        </button>
      </div>
    </div>

    <!-- Active Samples Inventory Breakdown -->
    <div class="mt-4 pt-3 border-t border-indigo-900/60 flex flex-wrap items-center justify-between gap-2 text-xs">
      <div class="flex items-center gap-2">
        <span class="text-indigo-300 font-semibold">Saved Samples on Disk:</span>
        <div id="samplesSummaryChips" class="flex flex-wrap gap-1.5">
          <span class="text-indigo-400/80 italic">Loading inventory...</span>
        </div>
      </div>
      <span id="recordStatusMsg" class="text-xs text-indigo-300 font-medium">Ready to record</span>
    </div>
  </section>
"""

if "id=\"recordSampleBtn\"" not in html:
    html = html.replace("</main>", "</main>\n" + studio_html)
    print("Added Studio UI below main!")

# 3. Add JS functions for Studio
studio_js = """
    // Studio Data Collection Variables
    let isRecordingSample = false;
    let recordedFramesBuffer = [];
    let recordCountdownTimer = null;
    let recordingTargetCount = 30;
    const recordingOverlay = document.getElementById('recordingOverlay');
    const recordingCountdownText = document.getElementById('recordingCountdownText');
    const recordingSubtext = document.getElementById('recordingSubtext');
    const recordingProgressBar = document.getElementById('recordingProgressBar');
    const recordLabelSelect = document.getElementById('recordLabelSelect');
    const customLabelInput = document.getElementById('customLabelInput');
    const recordSignerInput = document.getElementById('recordSignerInput');
    const recordSampleBtn = document.getElementById('recordSampleBtn');
    const recordStatusMsg = document.getElementById('recordStatusMsg');
    const sessionSamplesCount = document.getElementById('sessionSamplesCount');
    const samplesSummaryChips = document.getElementById('samplesSummaryChips');
    let sessionSavedCount = 0;

    function handleLabelSelectChange() {
      if (recordLabelSelect.value === '__custom__') {
        customLabelInput.classList.remove('hidden');
        customLabelInput.focus();
      } else {
        customLabelInput.classList.add('hidden');
      }
    }

    function getSelectedRecordLabel() {
      if (recordLabelSelect.value === '__custom__') {
        return (customLabelInput.value || 'custom_sign').trim().toLowerCase().replace(/\\s+/g, '_');
      }
      return recordLabelSelect.value.trim().toLowerCase().replace(/\\s+/g, '_');
    }

    async function refreshSamplesSummary() {
      try {
        const resp = await fetch('http://localhost:8000/api/samples_summary');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.status === 'ok') {
          sessionSamplesCount.textContent = data.total_samples || 0;
          samplesSummaryChips.innerHTML = '';
          const classes = Object.keys(data.samples_by_class || {});
          if (classes.length === 0) {
            samplesSummaryChips.innerHTML = '<span class=\"text-indigo-400/60\">None yet</span>';
          } else {
            classes.forEach(cls => {
              const chip = document.createElement('span');
              const isUnk = cls === 'unknown' || cls === 'no_sign';
              chip.className = isUnk 
                ? 'px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30 font-medium'
                : 'px-2 py-0.5 rounded-md bg-indigo-800/60 text-indigo-200 border border-indigo-700/40 font-medium';
              chip.textContent = `${cls}: ${data.samples_by_class[cls]}`;
              samplesSummaryChips.appendChild(chip);
            });
          }
        }
      } catch (e) {
        console.warn('Could not fetch samples summary:', e);
      }
    }

    async function startSampleRecording() {
      if (isRecordingSample) return;
      if (!mediaStream || !recognitionActive) {
        recordStatusMsg.textContent = 'Please start camera first!';
        recordStatusMsg.className = 'text-xs text-amber-400 font-semibold';
        alert('Please click \"Start Camera\" before recording gesture samples.');
        return;
      }

      const label = getSelectedRecordLabel();
      if (!label) {
        alert('Please enter or select a valid gesture label.');
        return;
      }

      recordSampleBtn.disabled = true;
      recordedFramesBuffer = [];
      recordingOverlay.classList.remove('hidden');
      recordingProgressBar.style.width = '0%';

      let countdown = 3;
      recordingCountdownText.textContent = countdown;
      recordingSubtext.textContent = `Get Ready for \"${label.toUpperCase()}\"...`;

      recordCountdownTimer = setInterval(() => {
        countdown -= 1;
        if (countdown > 0) {
          recordingCountdownText.textContent = countdown;
        } else {
          clearInterval(recordCountdownTimer);
          recordingCountdownText.textContent = '🔴';
          recordingSubtext.textContent = `RECORDING \"${label.toUpperCase()}\" (Perform Sign Now)...`;
          isRecordingSample = true;
        }
      }, 800);
    }

    async function finishSampleRecording() {
      isRecordingSample = false;
      recordingOverlay.classList.add('hidden');
      recordSampleBtn.disabled = false;
      const label = getSelectedRecordLabel();
      const signer = (recordSignerInput.value || 'signer').trim();

      if (recordedFramesBuffer.length < 15) {
        recordStatusMsg.textContent = 'Recording was too short; sample skipped.';
        recordStatusMsg.className = 'text-xs text-amber-400 font-medium';
        return;
      }

      recordStatusMsg.textContent = `Saving ${recordedFramesBuffer.length} frames for \"${label}\"...`;
      recordStatusMsg.className = 'text-xs text-indigo-300 font-medium';

      try {
        const resp = await fetch('http://localhost:8000/api/save_sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            label: label,
            signer: signer,
            frames: recordedFramesBuffer
          })
        });
        const res = await resp.json();
        if (res.status === 'ok') {
          sessionSavedCount += 1;
          recordStatusMsg.textContent = `✅ Saved sample for \"${res.label}\"! (Total for sign: ${res.total_for_class})`;
          recordStatusMsg.className = 'text-xs text-emerald-400 font-semibold';
          await refreshSamplesSummary();
        } else {
          recordStatusMsg.textContent = `Error: ${res.message}`;
          recordStatusMsg.className = 'text-xs text-red-400 font-medium';
        }
      } catch (err) {
        recordStatusMsg.textContent = `Backend save error: ${err.message}`;
        recordStatusMsg.className = 'text-xs text-red-400 font-medium';
      }
    }

    async function recordBurstIdleSamples() {
      recordLabelSelect.value = 'unknown';
      handleLabelSelectChange();
      for (let i = 1; i <= 3; i++) {
        recordStatusMsg.textContent = `Burst Take ${i}/3: Recording idle/resting pose...`;
        await new Promise(r => setTimeout(r, 600));
        startSampleRecording();
        while (isRecordingSample || !recordingOverlay.classList.contains('hidden')) {
          await new Promise(r => setTimeout(r, 200));
        }
        await new Promise(r => setTimeout(r, 1200));
      }
      recordStatusMsg.textContent = '✅ Completed 3-take Idle/Neutral calibration batch!';
      recordStatusMsg.className = 'text-xs text-emerald-400 font-semibold';
    }

    // Refresh summary on load
    setTimeout(refreshSamplesSummary, 1500);

    window.handleLabelSelectChange = handleLabelSelectChange;
    window.startSampleRecording = startSampleRecording;
    window.recordBurstIdleSamples = recordBurstIdleSamples;
"""

# In processVideoFrame, capture frames when isRecordingSample is true
old_process_end = "if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(Array.from(frame)));"
new_process_end = """if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(Array.from(frame)));

        // Buffer frame into sample collector if recording
        if (isRecordingSample) {
          recordedFramesBuffer.push(Array.from(frame));
          const pct = Math.min(100, Math.round((recordedFramesBuffer.length / recordingTargetCount) * 100));
          recordingProgressBar.style.width = `${pct}%`;
          if (recordedFramesBuffer.length >= recordingTargetCount) {
            finishSampleRecording();
          }
        }"""

if old_process_end in html:
    html = html.replace(old_process_end, new_process_end)
    print("Added frame capture to processVideoFrame!")

if "isRecordingSample = false" not in html:
    html = html.replace("window.copySentence = copySentence;", "window.copySentence = copySentence;\n" + studio_js)
    print("Added Studio JavaScript logic!")

with open(index_html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("index.html updated successfully!")
