import os

index_html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
with open(index_html_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update renderPrediction function
old_render = """    function renderPrediction(result) {
      const sign = String(result.predicted_sign || '').trim();
      const confidence = Math.max(0, Math.min(100, Number(result.confidence) || 0));
      if (!sign) return;
      predictionText.textContent = sign.toUpperCase();
      matchConfidence.textContent = `${confidence.toFixed(1)}% Match`;
      confidenceBar.style.width = `${confidence}%`;
      setStatus('Camera and live sign recognition active');

      if (confidence < 55) {
        candidateSign = '';
        candidateCount = 0;
        return;
      }
      if (candidateSign === sign) candidateCount += 1;
      else { candidateSign = sign; candidateCount = 1; }
      if (candidateCount >= 3 && committedSign !== sign) {
        const current = sentenceText.textContent.trim();
        sentenceText.textContent = current ? `${current} ${sign}` : sign;
        committedSign = sign;
        speakText(sign);
      }
    }"""

new_render = """    function renderPrediction(result) {
      const sign = String(result.predicted_sign || '').trim();
      const confidence = Math.max(0, Math.min(100, Number(result.confidence) || 0));
      if (!sign) return;
      
      const isUnknown = sign.toLowerCase() === 'unknown' || sign.toLowerCase() === 'no_sign';
      
      // Handle Unknown / Neutral pose
      if (isUnknown) {
        predictionText.textContent = 'Resting / Neutral';
        predictionText.className = 'text-2xl font-semibold tracking-tight text-gray-400 my-1';
        matchConfidence.textContent = 'No Sign Active';
        matchConfidence.className = 'bg-gray-100 text-gray-600 border border-gray-200 text-xs font-semibold px-2.5 py-0.5 rounded-full';
        confidenceBar.style.width = '0%';
        candidateSign = '';
        candidateCount = 0;
        return;
      }

      // Handle Low Confidence (< 60%) - Do not flash misleading guesses!
      if (confidence < 60) {
        predictionText.textContent = '—';
        predictionText.className = 'text-4xl font-black tracking-tight text-gray-300 my-1';
        matchConfidence.textContent = `${confidence.toFixed(1)}% (Low Confidence)`;
        matchConfidence.className = 'bg-amber-50 text-amber-700 border border-amber-200 text-xs font-semibold px-2.5 py-0.5 rounded-full';
        confidenceBar.style.width = `${confidence}%`;
        confidenceBar.className = 'bg-amber-400 h-full transition-all duration-300';
        candidateSign = '';
        candidateCount = 0;
        return;
      }

      // High Confidence valid sign
      predictionText.textContent = sign.toUpperCase();
      predictionText.className = 'text-4xl font-black tracking-tight text-emerald-600 my-1';
      matchConfidence.textContent = `${confidence.toFixed(1)}% Match`;
      matchConfidence.className = 'bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-semibold px-2.5 py-0.5 rounded-full';
      confidenceBar.style.width = `${confidence}%`;
      confidenceBar.className = 'bg-emerald-500 h-full transition-all duration-300';
      setStatus('Camera and live sign recognition active');

      if (candidateSign === sign) candidateCount += 1;
      else { candidateSign = sign; candidateCount = 1; }
      
      if (candidateCount >= 3 && committedSign !== sign) {
        const current = sentenceText.textContent.trim();
        const placeholder = 'Recognized signs will appear here.';
        sentenceText.textContent = (!current || current === placeholder) ? sign : `${current} ${sign}`;
        committedSign = sign;
        speakText(sign);
      }
    }"""

if old_render in html:
    html = html.replace(old_render, new_render)
    print("Replaced renderPrediction successfully!")
else:
    print("Warning: old_render exact string not found, checking normalized.")

# Save modified html
with open(index_html_path, "w", encoding="utf-8") as f:
    f.write(html)
