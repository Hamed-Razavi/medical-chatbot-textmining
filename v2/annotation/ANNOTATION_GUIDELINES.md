# Annotation guidelines (symptom validation)

Symptom annotation — 
Purpose
You are creating the reference (gold) labels used to measure how accurately an automatic extractor finds patient-reported symptoms. Work independently: do not discuss items with the other annotator until both files are finished.
Sheet "SampleA" (120 dialogues, about 90 minutes)
1. Read the whole dialogue, including the Bot turns, so you understand short replies such as "Yes, it does."
2. In column D, list every symptom the PATIENT (User) reports as currently PRESENT. Use concept names from the "Concepts" sheet, separated by semicolons, e.g.  fever; chills; sweating
3. Count a symptom when the patient confirms a Bot question ("Bot: Do you have chills?" / "User: Yes") — it is patient-reported.
4. Do NOT list: symptoms the patient denies ("no fever"); symptoms only the Bot mentions that the patient does not confirm; diseases or diagnoses (asthma, diabetes); medications; past symptoms that are clearly resolved.
5. If the patient reports a symptom that has no matching concept, write it in column E in the patient's words (e.g. hives; wheezy is wheezing).
6. Leave D empty if the patient reports no symptoms. Use column F for any comment or doubt.
Sheet "SampleB" (40 dialogues, about 25 minutes)
The extractor found no symptom in these dialogues. Choose ONE primary code in column D:
   S = the patient states symptoms in their own words, but the extractor missed them
   E = symptoms are reported only through short replies to Bot questions ("Yes, I do")
   M = the dialogue is about an existing diagnosis, medication, treatment or lifestyle, with no new symptoms
   O = other (explain in column E)
When both files are done, return them unchanged in structure (do not add or delete rows).

Clarification: count symptoms of the person the consultation is about, even if a parent or caregiver describes them; do not count family history ("my mother has diabetes").

## Concepts

| Concept | Includes |
|---|---|
| abdominal_pain | stomach pain/ache, cramps, belly pain |
| anal_swelling_lump | lump, swelling or pain around the anus |
| back_pain | back pain |
| bleeding | bleeding, blood in stool, nosebleeds |
| blisters | blisters, sores, oozing, crusts |
| bloating | bloated |
| blurred_vision | blurry vision, vision problems |
| chest_pain | chest pain, chest tightness |
| chills | chills, shivering |
| cloudy_urine | cloudy or strong-smelling urine |
| constipation | constipated, straining |
| cough | cough, coughing |
| dark_urine | dark urine |
| diarrhea | diarrhea, loose stools |
| dizziness | dizzy, lightheaded |
| excessive_thirst | very thirsty |
| eye_pain | pain behind the eyes |
| fatigue | tired, exhausted, weak, low energy |
| fever | fever, feverish, high temperature |
| frequent_urination | urinating often, many bathroom trips |
| headache | headache, head hurts, migraine pain |
| heartburn | heartburn, acid, burning in chest/throat, regurgitation |
| itching | itchy, itching |
| joint_pain | joint pain/swelling/stiffness, knee pain |
| leg_pain_heaviness | heavy, aching or tired legs |
| leg_veins | visible or bulging veins |
| light_sensitivity | sensitive to light or sound |
| loss_of_appetite | no appetite, not hungry |
| mucus | mucus, phlegm, sputum |
| muscle_pain | muscle aches, body aches |
| nasal_congestion | stuffy, congested, blocked nose |
| nausea | nausea, queasy |
| neck_pain | neck pain or stiffness |
| numbness_tingling | numbness, tingling, pins and needles |
| painful_urination | burning or pain when urinating |
| pale_stool | pale or clay-colored stool |
| pimples | pimples, acne, breakouts, blackheads |
| rash | rash, spots, red bumps, hives |
| runny_nose | runny nose |
| shortness_of_breath | short of breath, hard to breathe, breathless |
| skin_patches | patches, scaly, flaky skin |
| skin_redness | redness, inflamed skin |
| sneezing | sneezing |
| sore_throat | sore or scratchy throat |
| stiffness | stiffness (not tied to neck/joint) |
| sweating | sweating, night sweats |
| swelling | swelling, swollen, puffy (not joint) |
| vomiting | vomiting, throwing up |
| watery_eyes | watery, itchy or red eyes |
| weight_loss | losing weight |
| wheezing | wheezing, wheezy |
| yellow_skin_eyes | yellow skin or eyes (jaundice sense only) |
