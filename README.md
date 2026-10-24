# Sign Language Converter

A browser-based tool that reads signs from your webcam using **both hands**, shows the matching word on screen, builds a sentence, and speaks it aloud. No server, no Python, no installs.

## Features
- Tracks up to two hands in real time (MediaPipe Hand Landmarker)
- Built-in starter signs (numbers, Hello, Yes, Good, I love you, Namaste, ...)
- **Add your own signs from the website**: type a word, click Record, hold the sign
- Signs are saved in your browser and can be exported/imported as JSON
- Speaks words or the full sentence (English, Hindi, Malayalam voices if your device has them)

## Run it
The camera only works on `https` or `localhost`, and ES modules do not load from `file://`.

**Option A - GitHub Pages (recommended)**
1. Create a GitHub repo and upload all files in this folder to the root.
2. Go to Settings > Pages > Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
3. Open the `https://<your-username>.github.io/<repo>/` link.

**Option B - on your computer**
```
cd sign-language-converter
python -m http.server 8000
```
Then open http://localhost:8000

## Adding signs
1. Start the camera.
2. Type the word in "Add a new sign".
3. Click **Record**, wait for the 3-second countdown, and hold the sign for 3 seconds.
4. Record 2-3 times per sign, tilting or moving your hand slightly each time.
5. Use **Export** to back up your signs. To ship them with the site, you can import the file on any device.

## Important limits
- The built-in signs are simple hand shapes and are **not an official ISL dictionary**. Many real ISL signs involve movement, facial expression and body position. This app recognises hand shapes and positions at a moment in time, so it works best for static signs. Record your own versions of ISL signs for the words you need.
- Accuracy improves with good light, a plain background, and several recorded samples per sign.
- The **Matching strictness** slider controls how closely a sign must match your recordings.

## Files
- `index.html` - page layout
- `style.css` - styling
- `app.js` - hand tracking, recognition, recording, speech
