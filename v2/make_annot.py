import json, re, random, pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
SEED=2026
df=pd.read_csv('medical_conversations.csv')[['conversations','disease']].dropna().reset_index(drop=True)
df['dialogue_id']=['D%03d'%i for i in range(len(df))]
LEX=json.load(open('../symptom_lexicon.json'))
EX={'fever':'fever, feverish, high temperature','chills':'chills, shivering','sweating':'sweating, night sweats','headache':'headache, head hurts, migraine pain','fatigue':'tired, exhausted, weak, low energy','dizziness':'dizzy, lightheaded','nausea':'nausea, queasy','vomiting':'vomiting, throwing up','diarrhea':'diarrhea, loose stools','constipation':'constipated, straining','abdominal_pain':'stomach pain/ache, cramps, belly pain','bloating':'bloated','heartburn':'heartburn, acid, burning in chest/throat, regurgitation','loss_of_appetite':'no appetite, not hungry','weight_loss':'losing weight','cough':'cough, coughing','sore_throat':'sore or scratchy throat','nasal_congestion':'stuffy, congested, blocked nose','runny_nose':'runny nose','sneezing':'sneezing','shortness_of_breath':'short of breath, hard to breathe, breathless','wheezing':'wheezing, wheezy','chest_pain':'chest pain, chest tightness','mucus':'mucus, phlegm, sputum','joint_pain':'joint pain/swelling/stiffness, knee pain','muscle_pain':'muscle aches, body aches','neck_pain':'neck pain or stiffness','back_pain':'back pain','stiffness':'stiffness (not tied to neck/joint)','numbness_tingling':'numbness, tingling, pins and needles','rash':'rash, spots, red bumps, hives','itching':'itchy, itching','skin_patches':'patches, scaly, flaky skin','blisters':'blisters, sores, oozing, crusts','pimples':'pimples, acne, breakouts, blackheads','skin_redness':'redness, inflamed skin','swelling':'swelling, swollen, puffy (not joint)','yellow_skin_eyes':'yellow skin or eyes (jaundice sense only)','dark_urine':'dark urine','pale_stool':'pale or clay-colored stool','painful_urination':'burning or pain when urinating','frequent_urination':'urinating often, many bathroom trips','cloudy_urine':'cloudy or strong-smelling urine','excessive_thirst':'very thirsty','blurred_vision':'blurry vision, vision problems','bleeding':'bleeding, blood in stool, nosebleeds','leg_veins':'visible or bulging veins','leg_pain_heaviness':'heavy, aching or tired legs','watery_eyes':'watery, itchy or red eyes','light_sensitivity':'sensitive to light or sound','eye_pain':'pain behind the eyes','anal_swelling_lump':'lump, swelling or pain around the anus'}
assert set(EX)==set(LEX), set(LEX)^set(EX)
def fmt(c):
    turns=[t.strip() for t in c.split('</s>') if t.strip()]
    return '\n'.join(turns)
# Sample A: 5 per disease, stratified
rng=np.random.RandomState(SEED)
A=df.groupby('disease').sample(5,random_state=SEED).sample(frac=1,random_state=SEED)
# Sample B: 40 dialogues where frozen v1 lexicon found no concept, excluding A
sx=pd.read_csv('../symptoms_extracted.csv')
nocon=df[(sx.symptoms=='[]').values & ~df.dialogue_id.isin(A.dialogue_id)]
B=nocon.sample(40,random_state=SEED)
pd.DataFrame({'dialogue_id':A.dialogue_id,'disease':A.disease}).to_csv('annot/key_sampleA.csv',index=False)
pd.DataFrame({'dialogue_id':B.dialogue_id,'disease':B.disease}).to_csv('annot/key_sampleB.csv',index=False)
print('A',len(A),'B',len(B),'nocon pool',len(nocon), 'A no-concept', int((sx.symptoms=='[]').values[A.index].sum()))
thin=Side(style='thin',color='BFBFBF'); B4=Border(left=thin,right=thin,top=thin,bottom=thin)
HDR=PatternFill('solid',fgColor='DCE6F2'); INP=PatternFill('solid',fgColor='FFF9E5')
def build(annotator):
    wb=Workbook(); ws=wb.active; ws.title='Instructions'
    lines=[('Symptom annotation — '+annotator,True),('',False),
    ('Purpose',True),('You are creating the reference (gold) labels used to measure how accurately an automatic extractor finds patient-reported symptoms. Work independently: do not discuss items with the other annotator until both files are finished.',False),('',False),
    ('Sheet "SampleA" (120 dialogues, about 90 minutes)',True),
    ('1. Read the whole dialogue, including the Bot turns, so you understand short replies such as "Yes, it does."',False),
    ('2. In column D, list every symptom the PATIENT (User) reports as currently PRESENT. Use concept names from the "Concepts" sheet, separated by semicolons, e.g.  fever; chills; sweating',False),
    ('3. Count a symptom when the patient confirms a Bot question ("Bot: Do you have chills?" / "User: Yes") — it is patient-reported.',False),
    ('4. Do NOT list: symptoms the patient denies ("no fever"); symptoms only the Bot mentions that the patient does not confirm; diseases or diagnoses (asthma, diabetes); medications; past symptoms that are clearly resolved.',False),
    ('5. If the patient reports a symptom that has no matching concept, write it in column E in the patient\'s words (e.g. hives; wheezy is wheezing).',False),
    ('6. Leave D empty if the patient reports no symptoms. Use column F for any comment or doubt.',False),('',False),
    ('Sheet "SampleB" (40 dialogues, about 25 minutes)',True),
    ('The extractor found no symptom in these dialogues. Choose ONE primary code in column D:',False),
    ('   S = the patient states symptoms in their own words, but the extractor missed them',False),
    ('   E = symptoms are reported only through short replies to Bot questions ("Yes, I do")',False),
    ('   M = the dialogue is about an existing diagnosis, medication, treatment or lifestyle, with no new symptoms',False),
    ('   O = other (explain in column E)',False),('',False),
    ('When both files are done, return them unchanged in structure (do not add or delete rows).',True)]
    for i,(t,b) in enumerate(lines,1):
        c=ws.cell(i,1,t); c.font=Font(bold=b,size=12 if i==1 else 11); c.alignment=Alignment(wrap_text=True,vertical='top')
    ws.column_dimensions['A'].width=120
    wc=wb.create_sheet('Concepts'); wc.append(['Concept name (use exactly)','Includes, for example'])
    for k in sorted(EX): wc.append([k,EX[k]])
    for c in wc[1]: c.font=Font(bold=True); c.fill=HDR
    wc.column_dimensions['A'].width=24; wc.column_dimensions['B'].width=70
    def sheet(name,data,cols,widths,validation=None):
        s=wb.create_sheet(name); s.append(cols)
        for c in s[1]: c.font=Font(bold=True); c.fill=HDR; c.alignment=Alignment(wrap_text=True,vertical='center'); c.border=B4
        for n,(_,r) in enumerate(data.iterrows(),2):
            s.cell(n,1,n-1); s.cell(n,2,r.dialogue_id); s.cell(n,3,fmt(r.conversations))
            for j in range(1,len(cols)+1):
                cc=s.cell(n,j); cc.alignment=Alignment(wrap_text=True,vertical='top'); cc.border=B4
                if j>=4: cc.fill=INP
            s.row_dimensions[n].height=max(60,15*(fmt(r.conversations).count('\n')+1)+15*len(r.conversations)//110)
        for j,w in enumerate(widths,1): s.column_dimensions[get_column_letter(j)].width=w
        s.freeze_panes='D2'
        if validation:
            dv=DataValidation(type='list',formula1=validation,allow_blank=True); s.add_data_validation(dv); dv.add(f'D2:D{len(data)+1}')
    sheet('SampleA',A,['#','Dialogue ID','Dialogue (User = patient, Bot = chatbot)','Patient-reported symptoms (concept names; separated by ;)','Other symptoms not in concept list','Comment'],[5,10,80,40,25,25])
    sheet('SampleB',B,['#','Dialogue ID','Dialogue (User = patient, Bot = chatbot)','Primary code (S / E / M / O)','Comment'],[5,10,80,16,40],validation='"S,E,M,O"')
    wb.save(f'annot/Annotation_{annotator.replace(" ","_")}.xlsx')
build('Annotator A'); build('Annotator B')
