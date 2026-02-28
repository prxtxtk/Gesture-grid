// All ASL alphabet letters
const aslAlphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];

// === Game State ===
let currentQuestion = 0;
let score = 0; // Current quiz session score
let currentCorrectAnswer = "";
let currentAttempts = 0;
let currentQuestionType = 'multiple-choice';
let hintsGiven = 0;
let currentMode = 'random'; // 'random' or 'practice'
let usedLetters = [];
let availableLetters = [...aslAlphabet];
let referenceVisible = false;
let autoVideoStream = null;

// === XP & Persistence System ===
let userProfile = {
    totalXP: 0,
    level: 1,
    streak: 0,
    bestStreak: 0
};

// Hints Database
const signHints = {
    'A': ["Make a fist with your thumb on the side", "All fingers closed, thumb resting on side", "Palm facing forward"],
    'B': ["All fingers straight up, thumb across palm", "Four fingers up, thumb folded over palm", "Palm faces forward"],
    'C': ["Form a C shape with your hand", "Curve fingers like holding a cup", "Thumb and fingers make a curved C shape"],
    'D': ["Index finger up, others in fist", "Point with index, thumb on middle finger", "Looks like you're pointing"],
    'E': ["All fingers curled in, thumb over fingers", "Fingers bent like gripping a ball", "Palm faces you"],
    'F': ["Thumb and index finger touch, others up", "OK sign but with middle finger up", "Thumb and index make circle"],
    'G': ["Index finger pointing, thumb up", "Like pointing a gun", "Index finger extended, others in fist"],
    'H': ["Index and middle finger together pointing", "Two fingers extended like peace sign", "Fingers together, pointing sideways"],
    'I': ["Pinky finger up, others in fist", "Like little finger promise", "Pinky extended, thumb over other fingers"],
    'J': ["I sign but draw a J in air", "Pinky extended, move hand in J shape", "Make letter J motion with pinky"],
    'K': ["Index and middle up, thumb between", "Peace sign with thumb in middle", "Two fingers up, thumb between them"],
    'L': ["Index and thumb extended at right angle", "Like an L shape", "Thumb and index form 90 degree angle"],
    'M': ["Thumb under three fingers", "Fingers curved over thumb", "Like holding something small"],
    'N': ["Thumb under two fingers", "Middle and index over thumb", "Similar to M but with two fingers"],
    'O': ["Fingers curved to touch thumb", "Make an O shape", "All fingertips touching thumb"],
    'P': ["Index and thumb make circle, others up", "Like OK sign but pointing down", "Thumb and index circle, three fingers up"],
    'Q': ["Thumb and index make circle, pointing down", "Like P but pointing down", "OK sign pointing downward"],
    'R': ["Index and middle finger crossed", "Fingers crossed for luck", "Cross index over middle finger"],
    'S': ["Fist with thumb across fingers", "Like holding a ball", "Closed fist, thumb on top"],
    'T': ["Thumb between index and middle", "Fist with thumb peeking through", "Thumb tucked between fingers"],
    'U': ["Index and middle finger up together", "Peace sign pointing up", "Two fingers extended, close together"],
    'V': ["Index and middle finger spread apart", "Peace sign with fingers spread", "V for victory sign"],
    'W': ["Index, middle, ring finger up, spread", "Three fingers up making W shape", "Thumb and pinky touching"],
    'X': ["Index finger bent, others in fist", "Like pointing but finger bent", "Hook index finger"],
    'Y': ["Thumb and pinky extended", "Rock and roll sign", "Thumb and little finger out"],
    'Z': ["Index finger draws Z in air", "Trace Z shape with index", "Point and move in Z pattern"]
};

// DOM Elements
const scoreElement = document.getElementById('score-display');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const signImage = document.getElementById('current-sign');
const imageLoading = document.getElementById('image-loading');
const options = document.querySelectorAll('.option-card');
const feedback = document.getElementById('feedback');
const feedbackIcon = document.getElementById('feedback-icon');
const feedbackTitle = document.getElementById('feedback-title');
const feedbackMessage = document.getElementById('feedback-message');
const nextButton = document.getElementById('next-btn');
const userStreak = document.getElementById('user-streak');
const userXP = document.getElementById('user-xp');
const startLearningBtn = document.getElementById('start-learning');
const questionText = document.getElementById('question-text');
const signDisplay = document.getElementById('sign-display');

// XP DOM Elements
const levelBadge = document.getElementById('level-badge');
const xpFill = document.getElementById('xp-fill');
const levelUpOverlay = document.getElementById('level-up-overlay');
const newLevelNum = document.getElementById('new-level-num');

// Mode Selector Elements
const randomModeBtn = document.getElementById('random-mode-btn');
const practiceModeBtn = document.getElementById('practice-mode-btn');
const lettersSidebar = document.getElementById('letters-sidebar');
const lettersGrid = document.getElementById('letters-grid');

// Practice Reference Elements
const practiceReference = document.getElementById('practice-reference');
const referenceSign = document.getElementById('reference-sign');
const referenceLoading = document.getElementById('reference-loading');
const referenceLetterText = document.getElementById('reference-letter-text');
const referenceLetterDisplay = document.getElementById('reference-letter-display');
const referenceImageContainer = document.getElementById('reference-image-container');
const referenceToggleBtn = document.getElementById('reference-toggle-btn');

// Auto Video Interface Elements
const autoVideoInterface = document.getElementById('auto-video-interface');
const autoUserVideo = document.getElementById('auto-user-video');
const autoCaptureCanvas = document.getElementById('auto-capture-canvas');
const autoCaptureBtn = document.getElementById('auto-capture-btn');
const attemptsCount = document.getElementById('attempts-count');
const hintDisplay = document.getElementById('hint-display');
const showHintBtn = document.getElementById('show-hint-btn');
const autoVideoFeedback = document.getElementById('auto-video-feedback');

// Initialize the application
function initApp() {
    loadUserData(); // Load XP/Level from local storage
    createLettersGrid();
    updateUserStats();
    setupEventListeners();
    resetRandomMode();
    loadQuestion();

    startLearningBtn.addEventListener('click', () => {
        document.getElementById('learn').scrollIntoView({ behavior: 'smooth' });
    });
}

// === Persistence Functions ===
function loadUserData() {
    const savedData = localStorage.getItem('gestureGridData');
    if (savedData) {
        userProfile = JSON.parse(savedData);
    } else {
        userProfile = { totalXP: 0, level: 1, streak: 0, bestStreak: 0 };
    }
}

function saveUserData() {
    localStorage.setItem('gestureGridData', JSON.stringify(userProfile));
}

function calculateLevel(xp) {
    // Level formula: Level = Math.floor(XP / 100) + 1
    return Math.floor(xp / 100) + 1;
}

function gainXP(amount, sourceElement) {
    const oldLevel = userProfile.level;
    userProfile.totalXP += amount;
    userProfile.level = calculateLevel(userProfile.totalXP);
    
    // Animate visual XP gain
    if(sourceElement) {
        showFloatingXP(sourceElement, amount);
    }

    // Check Level Up
    if (userProfile.level > oldLevel) {
        showLevelUpModal(userProfile.level);
    }

    saveUserData();
    updateUserStats();
}

function showFloatingXP(element, amount) {
    const rect = element.getBoundingClientRect();
    const floatEl = document.createElement('div');
    floatEl.className = 'floating-xp';
    floatEl.textContent = `+${amount} XP`;
    floatEl.style.left = `${rect.left + rect.width / 2}px`;
    floatEl.style.top = `${rect.top}px`;
    document.body.appendChild(floatEl);

    setTimeout(() => {
        floatEl.remove();
    }, 1000);
}

function showLevelUpModal(level) {
    newLevelNum.textContent = level;
    levelUpOverlay.style.display = 'flex';
}

function createLettersGrid() {
    lettersGrid.innerHTML = '';
    aslAlphabet.forEach((letter) => {
        const letterCard = document.createElement('div');
        letterCard.className = 'letter-card';
        letterCard.innerHTML = `<div class="letter-display">${letter}</div><div class="letter-name">${letter}</div>`;
        letterCard.addEventListener('click', () => {
            if (currentMode === 'practice') {
                const letterIndex = aslAlphabet.indexOf(letter);
                if (letterIndex !== -1) {
                    currentQuestion = letterIndex;
                    loadQuestion();
                    updateLettersGrid();
                }
            }
        });
        lettersGrid.appendChild(letterCard);
    });
}

function updateLettersGrid() {
    const cards = document.querySelectorAll('.letter-card');
    cards.forEach((card, index) => {
        card.classList.toggle('active', index === currentQuestion);
    });
}

function resetRandomMode() {
    usedLetters = [];
    availableLetters = [...aslAlphabet].sort(() => Math.random() - 0.5);
    currentQuestion = 0;
}

function getRandomLetter() {
    if (availableLetters.length === 0) resetRandomMode();
    const randomLetter = availableLetters.pop();
    usedLetters.push(randomLetter);
    return aslAlphabet.indexOf(randomLetter);
}

// Update UI stats including XP bar
function updateUserStats() {
    userStreak.textContent = userProfile.streak;
    userXP.textContent = userProfile.totalXP;
    levelBadge.textContent = userProfile.level;
    scoreElement.textContent = `Score: ${score}`;

    // Calculate progress to next level
    const xpForNextLevel = userProfile.level * 100;
    const xpForCurrentLevel = (userProfile.level - 1) * 100;
    const levelProgress = userProfile.totalXP - xpForCurrentLevel;
    const levelRange = xpForNextLevel - xpForCurrentLevel;
    const percentage = Math.min(100, Math.max(0, (levelProgress / levelRange) * 100));
    
    xpFill.style.width = `${percentage}%`;
}

function setupEventListeners() {
    options.forEach(button => {
        button.addEventListener('click', (e) => checkAnswer(button.dataset.letter, e.currentTarget));
    });

    nextButton.addEventListener('click', nextQuestion);
    randomModeBtn.addEventListener('click', () => switchMode('random'));
    practiceModeBtn.addEventListener('click', () => switchMode('practice'));
    referenceToggleBtn.addEventListener('click', toggleReference);
    autoCaptureBtn.addEventListener('click', handleAutoCapture);
    showHintBtn.addEventListener('click', giveHint);

    document.addEventListener('keydown', (event) => {
        if (feedback.style.display !== 'flex') {
            if (event.key >= '1' && event.key <= '4') {
                const index = parseInt(event.key) - 1;
                if (options[index] && !options[index].disabled) {
                    checkAnswer(options[index].dataset.letter, options[index]);
                }
            }
        } else if (event.key === 'Enter') {
            nextQuestion();
        }
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelector(link.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
        });
    });
}

function toggleReference() {
    referenceVisible = !referenceVisible;
    if (referenceVisible) {
        referenceImageContainer.style.display = 'flex';
        referenceToggleBtn.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Reference';
    } else {
        referenceImageContainer.style.display = 'none';
        referenceToggleBtn.innerHTML = '<i class="fas fa-eye"></i> Show Reference';
    }
}

function switchMode(mode) {
    currentMode = mode;
    randomModeBtn.classList.toggle('active', mode === 'random');
    practiceModeBtn.classList.toggle('active', mode === 'practice');
    lettersSidebar.style.display = mode === 'practice' ? 'block' : 'none';
    document.querySelector('.main-content').style.marginLeft = mode === 'practice' ? '280px' : '0';
    practiceReference.style.display = mode === 'practice' ? 'flex' : 'none';
    referenceVisible = false;
    referenceImageContainer.style.display = 'none';
    referenceToggleBtn.innerHTML = '<i class="fas fa-eye"></i> Show Reference';
    
    if (mode === 'random') {
        resetRandomMode();
        score = 0;
    }
    loadQuestion();
}

function loadQuestion() {
    let currentLetter;
    if (currentMode === 'random') {
        const letterIndex = getRandomLetter();
        currentLetter = aslAlphabet[letterIndex];
        currentQuestion = letterIndex;
    } else {
        currentLetter = aslAlphabet[currentQuestion];
        updatePracticeReference(currentLetter);
    }
    
    currentAttempts = 0;
    hintsGiven = 0;
    
    if (currentMode === 'random') {
        currentQuestionType = Math.random() < 0.8 ? 'signing' : 'multiple-choice';
    } else {
        currentQuestionType = 'signing';
    }
    
    if (currentQuestionType === 'multiple-choice') {
        imageLoading.style.display = 'flex';
        signImage.style.display = 'none';
        signImage.src = `images/${currentLetter}.jpg`;
        signImage.alt = `ASL sign for ${currentLetter}`;
        
        const loadTimeout = setTimeout(() => {
            if (imageLoading.style.display !== 'none') {
                imageLoading.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Image not available';
                signImage.style.display = 'none';
                setupQuestion(currentLetter);
            }
        }, 3000);

        signImage.onload = function() {
            clearTimeout(loadTimeout);
            imageLoading.style.display = 'none';
            signImage.style.display = 'block';
            setupQuestion(currentLetter);
        };
        signImage.onerror = function() {
            clearTimeout(loadTimeout);
            imageLoading.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Image not available';
            signImage.style.display = 'none';
            setupQuestion(currentLetter);
        };
    } else {
        setupQuestion(currentLetter);
    }
}

function updatePracticeReference(currentLetter) {
    referenceLoading.style.display = 'flex';
    referenceSign.style.display = 'none';
    referenceLetterText.textContent = currentLetter;
    referenceLetterDisplay.textContent = currentLetter;
    referenceSign.src = `images/${currentLetter}.jpg`;
    
    const loadTimeout = setTimeout(() => {
        if (referenceLoading.style.display !== 'none') {
            referenceLoading.innerHTML = '<i class="fas fa-exclamation-triangle"></i> N/A';
        }
    }, 3000);

    referenceSign.onload = function() {
        clearTimeout(loadTimeout);
        referenceLoading.style.display = 'none';
        referenceSign.style.display = 'block';
    };
    referenceSign.onerror = function() {
        clearTimeout(loadTimeout);
        referenceLoading.innerHTML = '<i class="fas fa-exclamation-triangle"></i> N/A';
        referenceSign.style.display = 'none';
    };
}

function setupQuestion(currentLetter) {
    currentCorrectAnswer = currentLetter;
    if (currentQuestionType === 'signing') {
        setupSigningQuestion(currentLetter);
    } else {
        setupMultipleChoiceQuestion(currentLetter);
    }
    updateProgress();
    feedback.style.display = 'none';
    if (currentMode === 'practice') updateLettersGrid();
}

function setupSigningQuestion(currentLetter) {
    document.querySelector('.options-section').style.display = 'none';
    signDisplay.style.display = 'none';
    questionText.textContent = `Sign the letter: ${currentLetter}`;
    showAutoVideoInterface(currentLetter);
}

function setupMultipleChoiceQuestion(currentLetter) {
    document.querySelector('.options-section').style.display = 'block';
    signDisplay.style.display = 'block';
    questionText.textContent = 'What letter is this ASL sign?';
    autoVideoInterface.style.display = 'none';
    if (autoVideoStream) {
        autoVideoStream.getTracks().forEach(track => track.stop());
        autoVideoStream = null;
    }
    const wrongOptions = aslAlphabet.filter(letter => letter !== currentLetter);
    const shuffledWrongs = wrongOptions.sort(() => Math.random() - 0.5).slice(0, 3);
    const allOptions = [currentLetter, ...shuffledWrongs].sort(() => Math.random() - 0.5);

    options.forEach((button, index) => {
        const optionLetter = allOptions[index];
        button.querySelector('.option-letter').textContent = optionLetter;
        button.dataset.letter = optionLetter;
        button.disabled = false;
        button.style.borderColor = '';
        button.style.background = '';
    });
}

function showAutoVideoInterface(currentLetter) {
    autoVideoInterface.style.display = 'block';
    currentAttempts = 0;
    hintsGiven = 0;
    attemptsCount.textContent = currentAttempts;
    hintDisplay.textContent = '';
    showHintBtn.style.display = 'none';
    autoVideoFeedback.textContent = '';
    autoVideoFeedback.className = 'auto-video-feedback';
    startAutoVideo();
}

function startAutoVideo() {
    if (autoVideoStream) autoVideoStream.getTracks().forEach(track => track.stop());
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        .then(stream => {
            autoVideoStream = stream;
            autoUserVideo.srcObject = stream;
        })
        .catch(err => {
            console.error("Camera error", err);
            autoVideoFeedback.textContent = "Camera access denied.";
            autoVideoFeedback.className = 'auto-video-feedback feedback-incorrect';
        });
}

async function handleAutoCapture() {
    const imageData = captureAutoFrame();
    const result = await sendForPrediction(imageData);

    if (!result.success) {
        autoVideoFeedback.textContent = result.error === 'no_hand_detected' ? "No hand detected." : "Error analyzing.";
        autoVideoFeedback.className = 'auto-video-feedback feedback-incorrect';
        return;
    }

    currentAttempts++;
    attemptsCount.textContent = currentAttempts;
    
    const pred = String(result.pred).toUpperCase();
    const conf = result.confidence ? Number(result.confidence) : 0;

    // === NEW LOGIC: Construct detailed stats string regardless of result ===
    let feedbackDetails = `Detected: ${pred}`;
    if (conf > 0) {
        feedbackDetails += ` (${(conf * 100).toFixed(1)}%)`;
    }
    
    if (result.top && Array.isArray(result.top) && result.top.length > 0) {
        const top3 = result.top.slice(0, 3)
            .map(t => `${t[0]} (${(t[1]*100).toFixed(0)}%)`)
            .join(", ");
        feedbackDetails += ` | Candidates: ${top3}`;
    }

    if (pred === currentCorrectAnswer) {
        // CORRECT
        autoVideoFeedback.textContent = `✅ Correct! ${feedbackDetails}`;
        autoVideoFeedback.className = 'auto-video-feedback feedback-correct';
        score++;
        userProfile.streak++;
        
        // XP GAIN for Webcam
        gainXP(15, autoCaptureBtn); 
        
        showFeedback('correct', `✅ Perfect! You signed ${currentCorrectAnswer} correctly!`);
        
        if (currentMode === 'random') setTimeout(nextQuestion, 2000);

    } else {
        // INCORRECT
        autoVideoFeedback.textContent = `❌ Incorrect. ${feedbackDetails}`;
        autoVideoFeedback.className = 'auto-video-feedback feedback-incorrect';
        userProfile.streak = 0;
        updateUserStats();
        saveUserData(); // Save reset streak
        
        if (currentAttempts >= 1) showHintBtn.style.display = 'inline-flex';
        if (currentAttempts >= 4 && currentMode === 'random') showAnswerReveal();
    }
}

function captureAutoFrame() {
    const ctx = autoCaptureCanvas.getContext('2d');
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(autoUserVideo, -autoCaptureCanvas.width, 0, autoCaptureCanvas.width, autoCaptureCanvas.height);
    ctx.restore();
    return autoCaptureCanvas.toDataURL('image/png');
}

function giveHint() {
    if (hintsGiven < 3) {
        const hint = signHints[currentCorrectAnswer][hintsGiven];
        hintDisplay.textContent = `💡 Hint ${hintsGiven + 1}/3: ${hint}`;
        hintsGiven++;
        if (hintsGiven >= 3) showHintBtn.style.display = 'none';
    }
}

function showAnswerReveal() {
    const revealModal = document.createElement('div');
    revealModal.className = 'answer-reveal-modal';
    revealModal.innerHTML = `
        <h3>Time to learn!</h3>
        <p>Correct sign: <strong>${currentCorrectAnswer}</strong></p>
        <img src="images/${currentCorrectAnswer}.jpg" class="answer-reveal-image" onerror="this.style.display='none'">
        <button class="cta-button" onclick="this.parentElement.remove(); nextQuestion();">Continue</button>
    `;
    document.body.appendChild(revealModal);
}

// Send image to server for prediction (127.0.0.1)
async function sendForPrediction(dataUrl) {
  try {
    autoVideoFeedback.textContent = "Analyzing...";
    autoCaptureBtn.disabled = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    
    const resp = await fetch("http://127.0.0.1:5000/predict", {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ image: dataUrl }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    
    if (!resp.ok) return { success: false, error: "server_error" };
    const result = await resp.json();
    return result || { success: false, error: "invalid_response" };
  } catch (e) {
    return { success: false, error: "network" };
  } finally {
    autoCaptureBtn.disabled = false;
  }
}

function updateProgress() {
    if (currentMode === 'random') {
        const progress = (usedLetters.length / aslAlphabet.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Question ${usedLetters.length} of ${aslAlphabet.length}`;
    } else {
        progressFill.style.width = `0%`;
        progressText.textContent = `Practice Mode - Letter ${aslAlphabet[currentQuestion]}`;
    }
}

function checkAnswer(selectedLetter, buttonElement) {
    const isCorrect = selectedLetter === currentCorrectAnswer;

    options.forEach(button => {
        button.disabled = true;
        if (button.dataset.letter === currentCorrectAnswer) {
            button.style.borderColor = '#4CAF50';
            button.style.background = '#f8fff9';
        }
        if (button.dataset.letter === selectedLetter && !isCorrect) {
            button.style.borderColor = '#f72585';
            button.style.background = '#fff5f8';
        }
    });

    if (isCorrect) {
        score++;
        userProfile.streak++;
        // XP GAIN for Multiple Choice (Standard reward)
        gainXP(10, buttonElement);
        showFeedback('correct', `✅ Correct! It's letter ${currentCorrectAnswer}`);
    } else {
        userProfile.streak = 0;
        saveUserData(); // Save reset streak
        updateUserStats();
        showFeedback('incorrect', `❌ Incorrect! It was letter ${currentCorrectAnswer}`);
    }
}

function showFeedback(type, message) {
    feedback.className = `feedback-card ${type}`;
    feedbackTitle.textContent = type === 'correct' ? 'Excellent!' : 'Try Again!';
    feedbackMessage.textContent = message;
    feedbackIcon.className = type === 'correct' ? 'fas fa-check' : 'fas fa-times';
    feedback.style.display = 'flex';
}

function nextQuestion() {
    if (currentMode === 'random') {
        if (usedLetters.length >= aslAlphabet.length) {
            showCompletionModal();
            resetRandomMode();
            score = 0;
        } else {
            loadQuestion();
        }
    }
}

function showCompletionModal() {
    // XP Bonus for finishing a quiz
    gainXP(50, null); 
    const completionMessage = `🎉 Quiz Complete!\n\nScore: ${score}/${aslAlphabet.length}\nLevel: ${userProfile.level}\nTotal XP: ${userProfile.totalXP}\n\n+50 Bonus XP for finishing!`;
    if (confirm(completionMessage)) {
        updateUserStats();
    }
}

// initialize
window.addEventListener('load', initApp);