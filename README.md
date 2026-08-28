# Ripple

A platform where you can clone UGC videos, swap the character, voicing, or products
and generate clips up to a minute+ long. Uses a cut detection tool to break up your
video into clips and generate each individually. Review and approve each clip to be
stitched back together. 

## Installation

Clone the repo, then install the working environment and variables:

```bash
python3.11 -m venv .venv
  .venv/bin/python -m pip install -e '.[features]'

  mkdir -p .cutdetect/models
  curl --fail --location \

    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    \
    --output .cutdetect/models/face_landmarker.task

  .venv/bin/cutdetect pipeline studio
```

Then open `[http://127.0.0.1:8787/](http://127.0.0.1:8790)`

For the existing local installation, start up Ripple anytime with:

```bash
cd "YOUR-PROJECT-LOCATION"
.venv311/bin/cutdetect studio
```

## API Keys Setup

You need to add a Runway Dev API (must purchase credits) and Elevenlabs API (FREE).

Ripple:
1. Sign in to the [Runway Developer Portal](https://dev.runwayml.com/) and create an API
  organization.
  2. Open **API Keys**, create a descriptively named key, and copy it immediately—it is shown only
  once.
  3. Add credits from the organization’s **Billing** page. AI generation is paid, so consider
  setting a spending limit.
  4. Add the key to your local `.env` file:

     ```dotenv
     RUNWAYML_API_SECRET=your_key_here

ElevenLabs: (optional)
  1. In ElevenLabs, open **Developers → API Keys** and create a key.
  2. Keep **Restrict Key** enabled and allow **Speech to Text** access. Ripple does not need Text
  to Speech, Voices, Voice Cloning, or other permissions.
  3. Copy the key when it is created and add it to your local `.env` file:

     ```dotenv
     ELEVENLABS_API_KEY=your_key_here

Edit `.env.example` and remove the '.example', then add secret values ONLY to `.env`.
Never commit `.env`, paste keys into documentation and keep these private.

## Ripple Flow

Once you install it correctly and add the API keys, you can start testing Ripple. You can upload any
UGC reference video, and then your target character, product, and choose your target voice (ElevenLabs avatars only).
You can also try and upload your own avatar voice, but be warned it is a little buggy sometimes.

1: Once the inputs are uploaded, the default is 9:16 aspect ratio with 720p resolution, with Seedance 2.0 as the video model.
These are the only options for now. Once you check that you 'have permission to use these reference materials.', you can hit Generate.

2: The cut detector tool immediately takes effect, cutting a video into smaller pieces at the break if it's over 15 seconds. This step
will take time. Once it is done clipping them, it will immediately start video generation. NOTE: If you purchase $50 or more of 
Runway credits, you will be able to run up to 3 parallel generations at a time. If not, they will go 1 by 1 and take a bit longer.

3: After generation is finished (est. 5-10 min), you will be taken to the **Review** stage. Here you can view each individual clip 
generation and *approve*, *retry*, or *trim* the given clip. Once all clips are approved, press **Assemble** for the final result

<img width="2880" height="1800" alt="Ripple" src="https://github.com/user-attachments/assets/cd26795c-620d-44bc-afb5-a4f1d194cd19" />

4: After assembling, Ripple will stitch all of the clips together for the final output. You can download or preview your final video.

<img width="2880" height="1800" alt="Ripple1" src="https://github.com/user-attachments/assets/7c75ea52-946a-4b34-a460-f6167c443531" />

## Congrats!

For reference this is how much it costs on average for a video generation without retries:

<img width="598" height="310" alt="Screenshot 2026-08-27 at 8 08 11 PM" src="https://github.com/user-attachments/assets/e7bc54ae-a3f0-4083-a508-2cb733da580b" />

DM me your results @x.com/jtapp_99 if you want, thanks!
