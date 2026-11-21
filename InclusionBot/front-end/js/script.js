// SPED InclusionBot - chat page script
// The conversation is only kept on this page. Nothing is saved,
// so reloading the page clears the chat.
// Only the user's settings (text size, etc.) are saved on their device.

const MAX_TURNS = 8; // same limit as backend/app.py

// Get the parts of the page we need
const chatArea = document.getElementById("chat-area");
const messagesBox = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const sendLabel = sendBtn.querySelector(".send-label");
const resetBtn = document.getElementById("reset");

// Sidebar and pop-ups
const collapseBtn = document.getElementById("collapse-btn");
const menuBtn = document.getElementById("menu-btn");
const overlay = document.getElementById("overlay");
const settingsBtn = document.getElementById("settings-btn");
const helpBtn = document.getElementById("help-btn");
const settingsDialog = document.getElementById("settings-dialog");
const helpDialog = document.getElementById("help-dialog");

// Settings inputs
const sizeOptions = document.querySelectorAll('input[name="text-size"]');
const contrastBox = document.getElementById("high-contrast");
const animationBox = document.getElementById("no-animations");

// Voice parts (microphone, reading aloud and their settings)
const micBtn = document.getElementById("mic-btn");
const voiceModeBtn = document.getElementById("voice-mode");
const voiceStatus = document.getElementById("voice-status");
const autoReadBox = document.getElementById("auto-read");
const voiceLangSelect = document.getElementById("voice-lang");
const voiceNameSelect = document.getElementById("voice-name");
const voiceRateInput = document.getElementById("voice-rate");
const voiceRateValue = document.getElementById("voice-rate-value");
const voiceTestBtn = document.getElementById("voice-test");
const voiceSupportNote = document.getElementById("voice-support");

// Save the welcome message so "New chat" can show it again
const welcomeHTML = messagesBox.innerHTML;

const TYPING_DOTS = '<span class="typing"><span></span><span></span><span></span></span>';

// message sent when the user presses "Explain simply"
// (it mentions SPED so the bot does not think it is off-topic)
const SIMPLIFY_QUESTION = "Can you explain your last SPED answer again in simpler words?";

let chatHistory = [];   // past questions and answers sent to the backend
let controller = null;  // used to stop an answer (null = no answer loading)

let settings = {
  textSize: "normal",   // "normal", "large" or "xlarge"
  highContrast: false,
  noAnimations: false,
  autoRead: false,      // voice mode: read every answer aloud
  voiceLang: "auto",    // "auto", "en" or "fil"
  voiceName: "",        // "" = choose a voice automatically
  voiceRate: 0.85,      // reading speed, 0.6 to 1.4 (a little slower than normal)
  settingsVersion: 2,   // 2 = the slower default speed
  saveChats: true       // keep past chats in this browser (Recent chats)
};

// Chat history (saved in this browser only, never on the server)
const CHATS_KEY = "inclusionbot-chats";
const MAX_SAVED_CHATS = 30;
const historyList = document.getElementById("history-list");
const historyEmpty = document.getElementById("history-empty");
const saveChatsBox = document.getElementById("save-chats");
const deleteAllBtn = document.getElementById("delete-all-chats");

let chatLog = [];         // every question and answer of the chat on screen
let currentChatId = null; // id of the saved chat on screen (null = not saved yet)
let savedChats = [];      // [{ id, title, updated, messages }], newest first


// ---------- Helper functions ----------

// Turn special characters into safe text so they are not read as HTML
function escapeHTML(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Make **text** bold and *text* italic
function formatLine(text) {
  return escapeHTML(text)
    // only links to Philippine government sites (.gov.ph) become clickable
    .replace(/https:\/\/(?:[a-z0-9-]+\.)*gov\.ph[^\s<]*[^\s<.,;:!?)]/gi,
             '<a href="$&" target="_blank" rel="noopener noreferrer">$&</a>')
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\*(.+?)\*/g, "<i>$1</i>");
}

// Turn the bot's answer into HTML (paragraphs and lists)
function formatAnswer(text) {
  const lines = text.split("\n");
  let html = "";
  let openList = ""; // "", "ul" or "ol"

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    let listType = "";
    let itemText = "";

    if (line.startsWith("- ") || line.startsWith("* ") || line.startsWith("• ")) {
      listType = "ul";
      itemText = line.slice(2);
    } else if (/^\d+[.)] /.test(line)) {
      listType = "ol";
      itemText = line.replace(/^\d+[.)] /, "");
    }

    // close the old list if this line is not part of it
    if (openList !== "" && openList !== listType) {
      html += "</" + openList + ">";
      openList = "";
    }

    if (listType !== "") {
      if (openList === "") {
        html += "<" + listType + ">";
        openList = listType;
      }
      html += "<li>" + formatLine(itemText) + "</li>";
    } else if (line !== "") {
      html += "<p>" + formatLine(line.replace(/^#+\s*/, "")) + "</p>";
    }
  }

  if (openList !== "") {
    html += "</" + openList + ">";
  }
  return html;
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

// Add a chat bubble to the page and return the bubble
function addMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = "msg " + role;

  // only the bot has a picture beside its message
  if (role === "bot") {
    const avatar = document.createElement("img");
    avatar.className = "avatar";
    avatar.src = "images/bot-icon.png";
    avatar.alt = "";
    msg.appendChild(avatar);
  }

  // this box holds the bubble (and later the buttons under it)
  const body = document.createElement("div");
  body.className = "msg-body";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "user") {
    bubble.textContent = content; // user text is shown as plain text
  } else {
    bubble.innerHTML = content;
  }

  body.appendChild(bubble);
  msg.appendChild(body);
  messagesBox.appendChild(msg);
  scrollToBottom();
  return bubble;
}

// Add the buttons under a bot answer: Listen / Stop (reading aloud)
// and "Explain simply". Returns the Listen controls (or null).
function addAnswerButtons(bubble, reply, showSimplify) {
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  let reader = null;

  if (canSpeak) {
    reader = makeReadButtons(reply);
    actions.appendChild(reader.listenBtn);
    actions.appendChild(reader.stopBtn);
  }

  if (showSimplify) {
    const simpleBtn = document.createElement("button");
    simpleBtn.type = "button";
    simpleBtn.className = "action-btn simplify-btn";
    simpleBtn.textContent = "Explain simply";
    simpleBtn.addEventListener("click", function () {
      sendQuestion(SIMPLIFY_QUESTION);
    });
    actions.appendChild(simpleBtn);
  }

  bubble.parentElement.appendChild(actions);
  return reader;
}

// Switch the Send button to Stop while the bot is answering
function setBusy(busy) {
  if (busy) {
    sendBtn.classList.add("stop");
    sendLabel.textContent = "Stop";
    sendBtn.type = "button";
    input.disabled = true;
    micBtn.disabled = true;
  } else {
    sendBtn.classList.remove("stop");
    sendLabel.textContent = "Send";
    sendBtn.type = "submit";
    input.disabled = false;
    micBtn.disabled = !canListen;
    input.focus();
  }
}

// Make the text box grow taller when the user types more lines
function resizeInput() {
  input.style.height = "auto";
  // + 2 is for the 1px border on top and bottom
  input.style.height = Math.min(input.scrollHeight + 2, 160) + "px";

  // only show the scrollbar when the text is too long for the box
  if (input.scrollHeight > 160) {
    input.style.overflowY = "auto";
  } else {
    input.style.overflowY = "hidden";
  }
}


// ---------- Sending a question ----------

async function sendQuestion(question) {
  question = question.trim();
  if (question === "" || controller !== null) {
    return; // empty question, or still waiting for an answer
  }

  // a new question stops the microphone and any answer being read aloud
  cancelListening();
  stopReading();

  // hide the welcome screen once the conversation starts
  const welcome = document.getElementById("welcome");
  if (welcome) {
    welcome.remove();
  }

  // "Explain simply" is only for the newest answer, so remove the old ones
  const oldSimpleBtns = document.querySelectorAll(".simplify-btn");
  for (let i = 0; i < oldSimpleBtns.length; i++) {
    oldSimpleBtns[i].remove();
  }

  addMessage("user", question);
  input.value = "";
  resizeInput();

  const bubble = addMessage("bot", TYPING_DOTS);
  controller = new AbortController();
  setBusy(true);

  let reply = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: question,
        history: chatHistory.slice(-MAX_TURNS * 2)
      }),
      signal: controller.signal
    });

    if (!response.ok) {
      let errorMessage = "Something went wrong (error " + response.status + ").";
      try {
        const data = await response.json();
        if (data.error) {
          errorMessage = data.error;
        }
      } catch (e) {
        // the server did not send JSON, keep the default message
      }
      throw new Error(errorMessage);
    }

    // The server sends the answer little by little, so we read it in pieces
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const piece = await reader.read();
      if (piece.done) {
        break;
      }
      reply += decoder.decode(piece.value, { stream: true });
      bubble.innerHTML = formatAnswer(reply) || TYPING_DOTS;
      scrollToBottom();
    }
    reply += decoder.decode();

    if (reply.trim() === "") {
      bubble.innerHTML = "<p>(No answer was returned.)</p>";
    } else {
      bubble.innerHTML = formatAnswer(reply);
    }

  } catch (error) {
    if (error.name === "AbortError") {
      // the user pressed Stop
      bubble.innerHTML = formatAnswer(reply) + "<p><i>(Answer stopped.)</i></p>";
    } else {
      bubble.closest(".msg").classList.add("error");
      bubble.innerHTML = "<p>" + escapeHTML(error.message || "Could not reach the server.") + "</p>";
      reply = "";
    }
  }

  // save the question and answer so the bot remembers them next time
  // (but not if the user pressed "New chat" while it was still loading)
  if (reply.trim() !== "" && bubble.isConnected) {
    chatHistory.push({ role: "user", content: question });
    chatHistory.push({ role: "assistant", content: reply });
    chatHistory = chatHistory.slice(-MAX_TURNS * 2);

    // keep the whole chat so it can be opened again from "Recent chats"
    chatLog.push({ role: "user", content: question });
    chatLog.push({ role: "assistant", content: reply });
    rememberChat();

    // don't show "Explain simply" again if this answer is already the simple version
    const reader = addAnswerButtons(bubble, reply, question !== SIMPLIFY_QUESTION);

    // voice mode: read the finished answer aloud (not if the user pressed Stop)
    if (reader && settings.autoRead && !controller.signal.aborted) {
      reader.start();
    }
  }

  controller = null;
  setBusy(false);
  scrollToBottom();
}


// ---------- Settings ----------

// Change the look of the page based on the chosen settings
function applySettings() {
  document.body.classList.remove("text-large", "text-xlarge");
  if (settings.textSize === "large" || settings.textSize === "xlarge") {
    document.body.classList.add("text-" + settings.textSize);
  }
  document.body.classList.toggle("high-contrast", settings.highContrast);
  document.body.classList.toggle("no-animations", settings.noAnimations);
}

// Save the settings in the browser so they stay next time
function saveSettings() {
  try {
    localStorage.setItem("inclusionbot-settings", JSON.stringify(settings));
  } catch (e) {
    // some browsers block saving (for example private mode), that's okay
  }
}

// Load saved settings and show them in the Settings pop-up
function loadSettings() {
  try {
    const saved = localStorage.getItem("inclusionbot-settings");
    if (saved) {
      const savedSettings = JSON.parse(saved);
      settings.textSize = savedSettings.textSize || "normal";
      settings.highContrast = savedSettings.highContrast === true;
      settings.noAnimations = savedSettings.noAnimations === true;
      settings.autoRead = savedSettings.autoRead === true;
      settings.saveChats = savedSettings.saveChats !== false;
      if (["auto", "en", "fil"].includes(savedSettings.voiceLang)) {
        settings.voiceLang = savedSettings.voiceLang;
      }
      settings.voiceName = typeof savedSettings.voiceName === "string" ? savedSettings.voiceName : "";
      // speeds saved before the slower default (version 2) are replaced by the new default
      const rate = Number(savedSettings.voiceRate);
      if (savedSettings.settingsVersion >= 2 && rate >= 0.6 && rate <= 1.4) {
        settings.voiceRate = rate;
      }
    }
  } catch (e) {
    // nothing saved or it could not be read, keep the default settings
  }

  for (let i = 0; i < sizeOptions.length; i++) {
    sizeOptions[i].checked = (sizeOptions[i].value === settings.textSize);
  }
  contrastBox.checked = settings.highContrast;
  animationBox.checked = settings.noAnimations;
  autoReadBox.checked = settings.autoRead;
  saveChatsBox.checked = settings.saveChats;
  voiceLangSelect.value = settings.voiceLang;
  voiceRateInput.value = settings.voiceRate;
  showRate();
  showVoiceMode();
  loadVoices(); // shows the saved reading voice

  applySettings();
}

for (let i = 0; i < sizeOptions.length; i++) {
  sizeOptions[i].addEventListener("change", function () {
    settings.textSize = this.value;
    applySettings();
    saveSettings();
  });
}

contrastBox.addEventListener("change", function () {
  settings.highContrast = contrastBox.checked;
  applySettings();
  saveSettings();
});

animationBox.addEventListener("change", function () {
  settings.noAnimations = animationBox.checked;
  applySettings();
  saveSettings();
});


// ---------- Voice: reading answers aloud (text-to-speech) ----------
// Uses the browser's own speech voices. Nothing is sent to our server,
// and the answer text itself still comes only from the RAG backend.

const canSpeak = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
const canListen = Boolean(SpeechRecognitionAPI);

const ICON_PLAY = '<svg class="icon" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/></svg>';
const ICON_PAUSE = '<svg class="icon" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
const ICON_RESUME = '<svg class="icon" viewBox="0 0 24 24"><polygon points="6 4 20 12 6 20 6 4"/></svg>';
const ICON_STOP = '<svg class="icon" viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="1"/></svg>';

// common Filipino words, used to pick a Filipino voice for Filipino/Taglish answers
const FILIPINO_WORDS = ["po", "ang", "ng", "mga", "sa", "na", "ko", "mo", "ba", "para", "hindi",
  "paano", "anak", "siya", "ito", "kung", "may", "lang", "din", "rin", "natin", "nang", "opo",
  "salamat", "ninyo", "kayo", "namin", "ay", "pero", "kasi", "dahil"];

let reading = null;  // the answer being read now: { chunks, index, paused, token, ui }
let voices = [];     // voices the browser has

function setStatus(text) {
  voiceStatus.textContent = text;
}

function looksFilipino(text) {
  const words = text.toLowerCase().match(/[a-z]+/g) || [];
  let count = 0;
  for (let i = 0; i < words.length; i++) {
    if (FILIPINO_WORDS.includes(words[i])) {
      count++;
    }
  }
  return count >= 3 && count / words.length > 0.08;
}

// Which language to read in: "en" or "fil"
function readingLanguage(text) {
  if (settings.voiceLang === "auto") {
    return looksFilipino(text) ? "fil" : "en";
  }
  return settings.voiceLang;
}

function isFilipinoVoice(voice) {
  const lang = voice.lang.toLowerCase();
  return lang.startsWith("fil") || lang.startsWith("tl");
}

// Natural-sounding (neural) voices: Edge "Online (Natural)" voices, Google voices in Chrome,
// and newer Windows "Natural" voices. Older robotic voices (David, Zira...) score lowest.
function isNaturalVoice(voice) {
  return /natural|neural|online|google|premium|enhanced/i.test(voice.name);
}

function voiceScore(voice, lang, offlineOnly) {
  if (offlineOnly && !voice.localService) {
    return -1;
  }
  const code = voice.lang.toLowerCase();
  let score = 0;
  if (lang === "fil") {
    if (!isFilipinoVoice(voice)) {
      return -1;
    }
  } else {
    if (!code.startsWith("en")) {
      return -1;
    }
    if (code === "en-ph") {
      score += 3;         // Philippine English accent
    } else if (code === "en-us" || code === "en-gb") {
      score += 1;
    }
  }
  if (/natural|neural/i.test(voice.name)) {
    score += 10;          // the most human-sounding voices
  } else if (isNaturalVoice(voice)) {
    score += 6;
  }
  if (voice.default) {
    score += 0.5;
  }
  return score;
}

// The best voice for a language (offlineOnly = only voices installed on the device)
function bestVoice(lang, offlineOnly) {
  let best = null;
  let bestScore = -1;
  for (let i = 0; i < voices.length; i++) {
    const score = voiceScore(voices[i], lang, offlineOnly);
    if (score > bestScore) {
      best = voices[i];
      bestScore = score;
    }
  }
  return best;
}

// Pick the voice: the one chosen in Settings, otherwise the most natural one for the language
function pickVoice(lang) {
  if (settings.voiceName !== "") {
    const chosen = voices.find(function (v) { return v.name === settings.voiceName; });
    if (chosen) {
      return chosen;
    }
  }
  return bestVoice(lang, false) || bestVoice("en", false) || null;
}

// Pauses (in milliseconds) that make the reading sound like a person talking
const PAUSE_SENTENCE = 250;  // between sentences
const PAUSE_LINE = 450;      // after a heading, a bullet point or a paragraph

// Write short forms out in full so the voice says them naturally
function speakNaturally(text) {
  return text
    .replace(/https?:\/\/\S+/g, "the link on screen")  // never read a web address aloud
    .replace(/\be\.g\.,?/gi, "for example,")
    .replace(/\bi\.e\.,?/gi, "that is,")
    .replace(/\betc\./gi, "and so on.")
    .replace(/\bvs\.?\s/gi, "versus ")
    .replace(/\s&\s/g, " and ")
    .replace(/(\w)\/(\w)/g, "$1 or $2")   // "teacher/parent" -> "teacher or parent"
    .replace(/\s+[–—]\s+/g, ", ")          // dashes -> a short pause
    .replace(/\s{2,}/g, " ");
}

// Turn the answer (with **bold**, ### headings, bullets) into clean lines to read
function speakableLines(reply) {
  const main = reply.split(/\n\s*---\s*\n/)[0]; // ignore an old "Source" footer
  const lines = main.split("\n");
  const clean = [];
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim()
      .replace(/^#+\s*/, "")            // headings
      .replace(/^[-*•]\s+/, "")         // bullet marks
      .replace(/^\d+[.)]\s+/, "")       // numbered steps
      .replace(/\*\*(.+?)\*\*/g, "$1")  // bold
      .replace(/\*(.+?)\*/g, "$1")      // italics
      .replace(/\[learner\]/gi, "the learner")
      .trim();
    if (line === "") {
      continue;
    }
    line = speakNaturally(line);
    // end every line like a sentence so the voice lowers its tone and stops
    if (!/[.!?:;]$/.test(line)) {
      line += ".";
    }
    clean.push(line);
  }
  return clean;
}

// Split the answer into sentences, each with the pause that comes after it.
// Short pieces also stop some browsers from cutting off long text,
// and make Pause/Resume reliable.
function splitIntoChunks(lines) {
  const chunks = [];
  for (let i = 0; i < lines.length; i++) {
    const sentences = lines[i].match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) || [lines[i]];
    let current = "";
    for (let j = 0; j < sentences.length; j++) {
      current += sentences[j];
      // keep very short sentences together with the next one
      if (current.trim().length < 40 && j < sentences.length - 1) {
        continue;
      }
      chunks.push({ text: current.trim(), pause: j === sentences.length - 1 ? PAUSE_LINE : PAUSE_SENTENCE });
      current = "";
    }
  }
  return chunks.filter(function (c) { return c.text !== ""; });
}

// Make the Listen and Stop buttons for one answer
function makeReadButtons(reply) {
  const listenBtn = document.createElement("button");
  listenBtn.type = "button";
  listenBtn.className = "action-btn listen-btn";

  const stopBtn = document.createElement("button");
  stopBtn.type = "button";
  stopBtn.className = "action-btn stop-read-btn";
  stopBtn.innerHTML = ICON_STOP + "<span>Stop</span>";
  stopBtn.setAttribute("aria-label", "Stop reading");
  stopBtn.hidden = true;

  const ui = {
    listenBtn: listenBtn,
    stopBtn: stopBtn,
    start: function () {
      startReading(reply, ui);
    }
  };

  showReadState(ui, "idle");

  listenBtn.addEventListener("click", function () {
    if (reading && reading.ui === ui) {
      if (reading.paused) {
        resumeReading();
      } else {
        pauseReading();
      }
    } else {
      ui.start();
    }
  });
  stopBtn.addEventListener("click", function () {
    stopReading();
    listenBtn.focus();
  });

  return ui;
}

// Change the Listen button to show "Listen", "Pause" or "Resume"
function showReadState(ui, state) {
  if (state === "playing") {
    ui.listenBtn.innerHTML = ICON_PAUSE + "<span>Pause</span>";
    ui.listenBtn.setAttribute("aria-label", "Pause reading");
    ui.listenBtn.classList.add("active");
    ui.stopBtn.hidden = false;
  } else if (state === "paused") {
    ui.listenBtn.innerHTML = ICON_RESUME + "<span>Resume</span>";
    ui.listenBtn.setAttribute("aria-label", "Resume reading");
    ui.listenBtn.classList.add("active");
    ui.stopBtn.hidden = false;
  } else {
    ui.listenBtn.innerHTML = ICON_PLAY + "<span>Listen</span>";
    ui.listenBtn.setAttribute("aria-label", "Listen to this answer");
    ui.listenBtn.classList.remove("active");
    ui.stopBtn.hidden = true;
  }
}

function startReading(reply, ui) {
  if (!canSpeak) {
    return;
  }
  stopReading();
  stopListening();

  const lines = speakableLines(reply);
  if (lines.length === 0) {
    return;
  }
  const lang = readingLanguage(lines.join(" "));
  reading = {
    chunks: splitIntoChunks(lines),
    index: 0,
    paused: false,
    token: 0,
    lang: lang,
    voice: pickVoice(lang),
    ui: ui
  };
  showReadState(ui, "playing");
  speakNextChunk();
}

function speakNextChunk() {
  const r = reading;
  if (!r || r.paused) {
    return;
  }
  if (r.index >= r.chunks.length) {
    finishReading();
    return;
  }

  const chunk = r.chunks[r.index];
  const utterance = new SpeechSynthesisUtterance(chunk.text);
  if (r.voice) {
    utterance.voice = r.voice;
    utterance.lang = r.voice.lang;
  } else {
    utterance.lang = r.lang === "fil" ? "fil-PH" : "en-US";
  }
  utterance.rate = settings.voiceRate;
  utterance.pitch = 1;

  // the token tells us if this piece was cancelled by Pause or Stop
  const token = ++r.token;
  function next() {
    if (reading === r && r.token === token && !r.paused) {
      r.index++;
      // a short breath before the next sentence; slower reading = slightly longer pauses
      setTimeout(function () {
        if (reading === r && r.token === token && !r.paused) {
          speakNextChunk();
        }
      }, chunk.pause / settings.voiceRate);
    }
  }
  utterance.onend = next;
  utterance.onerror = function (event) {
    if (event.error === "interrupted" || event.error === "canceled") {
      return; // we stopped it ourselves
    }
    // natural online voices need internet: switch to a voice on this device and try again
    if (reading === r && r.token === token && r.voice && !r.voice.localService) {
      r.voice = bestVoice(r.lang, true) || bestVoice("en", true);
      speakNextChunk();
      return;
    }
    next();
  };
  window.speechSynthesis.speak(utterance);
}

// Pause: stop the voice but remember where we are, so Resume starts that sentence again
function pauseReading() {
  if (!reading || reading.paused) {
    return;
  }
  reading.paused = true;
  reading.token++;
  window.speechSynthesis.cancel();
  showReadState(reading.ui, "paused");
}

function resumeReading() {
  if (!reading || !reading.paused) {
    return;
  }
  reading.paused = false;
  showReadState(reading.ui, "playing");
  speakNextChunk();
}

function stopReading() {
  if (!canSpeak) {
    return;
  }
  const r = reading;
  reading = null;
  window.speechSynthesis.cancel();
  if (r) {
    r.token++;
    showReadState(r.ui, "idle");
  }
}

function finishReading() {
  const r = reading;
  reading = null;
  if (r) {
    showReadState(r.ui, "idle");
  }
}

// Fill the "Reading voice" list in Settings
function loadVoices() {
  if (!canSpeak) {
    return;
  }
  voices = window.speechSynthesis.getVoices().filter(function (v) {
    const lang = v.lang.toLowerCase();
    return lang.startsWith("en") || lang.startsWith("fil") || lang.startsWith("tl");
  });

  // natural voices first in the list
  voices.sort(function (a, b) {
    return Number(isNaturalVoice(b)) - Number(isNaturalVoice(a));
  });

  voiceNameSelect.innerHTML = '<option value="">Automatic (most natural voice)</option>';
  for (let i = 0; i < voices.length; i++) {
    const option = document.createElement("option");
    option.value = voices[i].name;
    option.textContent = (isNaturalVoice(voices[i]) ? "★ Natural · " : "")
      + voices[i].name + " (" + voices[i].lang + ")"
      + (voices[i].localService ? "" : " · needs internet");
    voiceNameSelect.appendChild(option);
  }
  // keep the saved voice if this device still has it
  const stillThere = voices.some(function (v) { return v.name === settings.voiceName; });
  voiceNameSelect.value = stillThere ? settings.voiceName : "";

  const notes = [];
  if (voices.length > 0 && !voices.some(isNaturalVoice)) {
    notes.push("This browser has no natural-sounding voices. For the most natural voices, "
      + "open InclusionBot in Microsoft Edge (or Chrome) while connected to the internet.");
  }
  if (voices.length > 0 && !voices.some(isFilipinoVoice)) {
    notes.push("No Filipino voice is available, so Filipino answers are read with an English voice. "
      + "Microsoft Edge has natural Filipino voices (Blessica, Angelo).");
  }
  voiceSupportNote.textContent = notes.join(" ");
}

function showRate() {
  voiceRateValue.textContent = Number(settings.voiceRate).toFixed(2) + "×";
}

// Show if voice mode is on or off on the Voice button
function showVoiceMode() {
  voiceModeBtn.setAttribute("aria-pressed", settings.autoRead ? "true" : "false");
  voiceModeBtn.title = settings.autoRead
    ? "Voice mode is on: answers are read aloud. Click to turn off."
    : "Voice mode is off. Click to read every answer aloud.";
}

function setVoiceMode(on) {
  settings.autoRead = on;
  autoReadBox.checked = on;
  showVoiceMode();
  saveSettings();
  if (!on) {
    stopReading();
  }
  setStatus(on ? "Voice mode on: answers will be read aloud." : "Voice mode off.");
}

voiceModeBtn.addEventListener("click", function () {
  setVoiceMode(!settings.autoRead);
});

autoReadBox.addEventListener("change", function () {
  setVoiceMode(autoReadBox.checked);
});

voiceLangSelect.addEventListener("change", function () {
  settings.voiceLang = voiceLangSelect.value;
  saveSettings();
});

voiceNameSelect.addEventListener("change", function () {
  settings.voiceName = voiceNameSelect.value;
  saveSettings();
});

voiceRateInput.addEventListener("input", function () {
  settings.voiceRate = Number(voiceRateInput.value);
  showRate();
  saveSettings();
});

voiceTestBtn.addEventListener("click", function () {
  const sample = settings.voiceLang === "fil"
    ? "Kumusta po! Ako si InclusionBot. Ganito po ang boses ko."
    : "Hello! I am InclusionBot. This is how I will read my answers.";
  startReading(sample, {
    listenBtn: document.createElement("button"),  // not shown on the page
    stopBtn: document.createElement("button")
  });
});


// ---------- Voice: speaking a question (speech-to-text) ----------
// The words appear in the question box. The user checks them and presses Send.

let recognizer = null;  // the microphone session, or null when it is off
let typedBefore = "";   // text that was already in the box before speaking

function showMicState(on) {
  micBtn.classList.toggle("listening", on);
  micBtn.setAttribute("aria-pressed", on ? "true" : "false");
  micBtn.setAttribute("aria-label", on ? "Stop listening" : "Speak your question");
  micBtn.title = on ? "Stop listening" : "Speak your question";
}

function startListening() {
  if (!canListen || recognizer || controller !== null) {
    return;
  }
  stopReading(); // don't let the bot's voice be heard by the microphone

  const r = new SpeechRecognitionAPI();
  r.lang = settings.voiceLang === "fil" ? "fil-PH" : "en-US";
  r.interimResults = true;   // show the words while the user is still speaking
  r.continuous = true;       // allow short pauses; press the button again to finish
  r.maxAlternatives = 1;

  typedBefore = input.value.trim() === "" ? "" : input.value.trim() + " ";
  let heardSomething = false;

  r.onresult = function (event) {
    if (recognizer !== r) {
      return; // this session was cancelled (for example the question was already sent)
    }
    let text = "";
    for (let i = 0; i < event.results.length; i++) {
      text += event.results[i][0].transcript;
    }
    heardSomething = text.trim() !== "";
    input.value = (typedBefore + text.trim()).slice(0, 2000);
    resizeInput();
  };

  r.onerror = function (event) {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      setStatus("The microphone is blocked. Allow microphone access for this page in your browser, then try again.");
    } else if (event.error === "no-speech") {
      setStatus("I didn't hear anything. Press the microphone and try again.");
    } else if (event.error === "network") {
      setStatus("Speaking a question needs an internet connection. You can still type your question.");
    } else if (event.error === "audio-capture") {
      setStatus("No microphone was found on this device.");
    } else if (event.error === "language-not-supported") {
      setStatus("This browser can't listen in that language. Try English in Settings › Voice.");
    } else if (event.error !== "aborted") {
      setStatus("The microphone stopped (" + event.error + "). Please try again.");
    }
  };

  r.onend = function () {
    if (recognizer !== r) {
      return; // already cancelled
    }
    recognizer = null;
    showMicState(false);
    if (heardSomething) {
      setStatus("Check your question, then press Send (or Enter).");
    }
    if (!input.disabled) {
      input.focus();
    }
  };

  try {
    r.start();
  } catch (e) {
    setStatus("The microphone could not start. Please try again.");
    return;
  }
  recognizer = r;
  showMicState(true);
  setStatus("Listening… speak your question. Press the microphone again when you are done.");
}

// Finish listening: the last words are still added to the box
function stopListening() {
  if (recognizer) {
    recognizer.stop(); // onend runs after the last words are added
  }
}

// Throw the microphone session away (when sending or starting a new chat)
function cancelListening() {
  if (recognizer) {
    const r = recognizer;
    recognizer = null;
    r.abort();
    showMicState(false);
  }
}

micBtn.addEventListener("click", function () {
  if (recognizer) {
    stopListening();
  } else {
    startListening();
  }
});

// Turn off what this browser can't do, and say why
if (!canListen) {
  micBtn.disabled = true;
  micBtn.title = "This browser can't turn speech into text. Try Chrome or Edge.";
  micBtn.setAttribute("aria-label", micBtn.title);
}
if (!canSpeak) {
  voiceModeBtn.hidden = true;
  document.querySelector(".voice-settings").hidden = true;
} else {
  loadVoices();
  window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
}

// Esc stops the microphone and the reading (pop-ups also close with Esc)
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape" && (recognizer || reading)) {
    stopListening();
    stopReading();
    setStatus("Stopped.");
  }
});


// ---------- Sidebar and pop-ups ----------

function closeMenu() {
  document.body.classList.remove("menu-open");
}

// computer: hide or show the sidebar
collapseBtn.addEventListener("click", function () {
  document.body.classList.toggle("sidebar-collapsed");
});

// phone: open the sidebar with the menu button
menuBtn.addEventListener("click", function () {
  document.body.classList.remove("sidebar-collapsed");
  document.body.classList.add("menu-open");
});

// phone: tapping outside the sidebar closes it
overlay.addEventListener("click", closeMenu);

settingsBtn.addEventListener("click", function () {
  closeMenu();
  settingsDialog.showModal();
});

helpBtn.addEventListener("click", function () {
  closeMenu();
  helpDialog.showModal();
});


// ---------- Chat events (clicks and key presses) ----------

form.addEventListener("submit", function (event) {
  event.preventDefault();
  sendQuestion(input.value);
});

// When the button says "Stop", clicking it stops the answer
sendBtn.addEventListener("click", function () {
  if (controller !== null) {
    controller.abort();
  }
});

// Enter = send, Shift + Enter = new line
input.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", resizeInput);

// Suggested question buttons
// (we put the click on the messages box because "New chat" makes new buttons)
messagesBox.addEventListener("click", function (event) {
  if (event.target.classList.contains("suggestion")) {
    sendQuestion(event.target.textContent);
  }
});

// "New chat" button clears everything
resetBtn.addEventListener("click", startNewChat);

// Clear the screen and stop everything that is running (used by New chat and opening a chat)
function clearChatScreen() {
  if (controller !== null) {
    controller.abort();
  }
  closeMenu();
  cancelListening();
  stopReading();
  setStatus("");
  chatHistory = [];
  chatLog = [];
  currentChatId = null;
  input.value = "";
  resizeInput();
}

function startNewChat() {
  clearChatScreen();
  messagesBox.innerHTML = welcomeHTML;
  renderHistory();
  input.focus();
}


// ---------- Chat history (Recent chats) ----------

// Read the saved chats from this browser
function readSavedChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(CHATS_KEY) || "[]");
    if (!Array.isArray(saved)) {
      return [];
    }
    return saved.filter(function (chat) {
      return chat && typeof chat.id === "string" && Array.isArray(chat.messages) && chat.messages.length > 0;
    });
  } catch (e) {
    return []; // nothing saved, or saving is blocked (private mode)
  }
}

// Write the saved chats; if the browser is full, drop the oldest chats and try again
function writeSavedChats() {
  while (true) {
    try {
      if (savedChats.length === 0) {
        localStorage.removeItem(CHATS_KEY);
      } else {
        localStorage.setItem(CHATS_KEY, JSON.stringify(savedChats));
      }
      return;
    } catch (e) {
      if (savedChats.length <= 1) {
        return; // saving is blocked, the chat still works without it
      }
      savedChats.pop();
    }
  }
}

// A short name for a chat: its first question
function chatTitle(messages) {
  const first = messages.find(function (m) { return m.role === "user"; });
  const text = first ? first.content.replace(/\s+/g, " ").trim() : "Chat";
  return text.length > 48 ? text.slice(0, 47) + "…" : text;
}

// Save the chat on screen (called after every answer)
function rememberChat() {
  if (!settings.saveChats || chatLog.length === 0) {
    return;
  }
  if (currentChatId === null) {
    currentChatId = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }
  const id = currentChatId;
  savedChats = savedChats.filter(function (chat) { return chat.id !== id; });
  savedChats.unshift({
    id: id,
    title: chatTitle(chatLog),
    updated: Date.now(),
    messages: chatLog.slice()
  });
  savedChats = savedChats.slice(0, MAX_SAVED_CHATS);
  writeSavedChats();
  renderHistory();
}

function formatChatDate(time) {
  const date = new Date(time);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return "Today, " + date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}

// Show the list of saved chats in the sidebar
function renderHistory() {
  historyList.innerHTML = "";

  if (!settings.saveChats) {
    historyEmpty.textContent = "Saving chats is off (Settings › Chat history).";
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.textContent = "Your chats will appear here.";
  historyEmpty.hidden = savedChats.length > 0;

  for (let i = 0; i < savedChats.length; i++) {
    const chat = savedChats[i];
    const item = document.createElement("li");
    item.className = "history-item";

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "history-open";
    openBtn.title = chat.title + " (" + formatChatDate(chat.updated) + ")";
    if (chat.id === currentChatId) {
      openBtn.setAttribute("aria-current", "true");
      item.classList.add("current");
    }
    const name = document.createElement("span");
    name.className = "history-name";
    name.textContent = chat.title;
    const date = document.createElement("span");
    date.className = "history-date";
    date.textContent = formatChatDate(chat.updated);
    openBtn.appendChild(name);
    openBtn.appendChild(date);
    openBtn.addEventListener("click", function () {
      openChat(chat.id);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "history-delete";
    deleteBtn.textContent = "×";
    deleteBtn.title = "Delete this chat";
    deleteBtn.setAttribute("aria-label", "Delete chat: " + chat.title);
    deleteBtn.addEventListener("click", function () {
      deleteChat(chat.id);
    });

    item.appendChild(openBtn);
    item.appendChild(deleteBtn);
    historyList.appendChild(item);
  }
}

// Show a saved chat again; the user can keep asking in it
function openChat(id) {
  const chat = savedChats.find(function (c) { return c.id === id; });
  if (!chat) {
    return;
  }
  clearChatScreen();
  messagesBox.innerHTML = "";
  currentChatId = chat.id;
  chatLog = chat.messages.slice();
  chatHistory = chatLog.slice(-MAX_TURNS * 2); // the bot remembers the end of this chat

  for (let i = 0; i < chatLog.length; i++) {
    const m = chatLog[i];
    if (m.role === "user") {
      addMessage("user", m.content);
    } else {
      const bubble = addMessage("bot", formatAnswer(m.content));
      // "Explain simply" only under the last answer, like in a live chat
      const isLast = i === chatLog.length - 1;
      const question = i > 0 ? chatLog[i - 1].content : "";
      addAnswerButtons(bubble, m.content, isLast && question !== SIMPLIFY_QUESTION);
    }
  }

  renderHistory();
  scrollToBottom();
  setStatus("Opened: " + chat.title);
  input.focus();
}

function deleteChat(id) {
  const chat = savedChats.find(function (c) { return c.id === id; });
  if (!chat || !window.confirm("Delete this chat?\n\n" + chat.title)) {
    return;
  }
  savedChats = savedChats.filter(function (c) { return c.id !== id; });
  writeSavedChats();
  if (id === currentChatId) {
    startNewChat();
  } else {
    renderHistory();
  }
  setStatus("Chat deleted.");
}

function deleteAllChats() {
  savedChats = [];
  writeSavedChats();
  startNewChat();
}

deleteAllBtn.addEventListener("click", function () {
  if (savedChats.length === 0) {
    window.alert("There are no saved chats.");
    return;
  }
  if (window.confirm("Delete all saved chats on this device? This cannot be undone.")) {
    deleteAllChats();
    setStatus("All saved chats were deleted.");
  }
});

saveChatsBox.addEventListener("change", function () {
  if (!saveChatsBox.checked) {
    if (savedChats.length > 0
        && !window.confirm("Turn off saving? The chats saved on this device will be deleted.")) {
      saveChatsBox.checked = true;
      return;
    }
    settings.saveChats = false;
    saveSettings();
    savedChats = [];
    writeSavedChats();
    currentChatId = null;
    renderHistory();
  } else {
    settings.saveChats = true;
    saveSettings();
    rememberChat(); // save the chat that is on screen now
    renderHistory();
  }
});


// ---------- Start ----------

loadSettings();
savedChats = settings.saveChats ? readSavedChats() : [];
renderHistory();
resizeInput();
input.focus();
