from typing import Tuple, List
from faker import Faker
# from datetime import datetime

import os
import json
import torch
import shutil
import asyncio
import autogen
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from autogen import OpenAIWrapper, AssistantAgent, UserProxyAgent

from pyhealth.medcode import InnerMap
from pyhealth.datasets import MIMIC3Dataset, SampleEHRDataset
from pyhealth.tasks import medication_recommendation_mimic3_fn
from pyhealth.models import GNN
from pyhealth.explainer import HeteroGraphExplainer

PATH = "output/20240124-084903/best.ckpt"
shutil.rmtree(".cache/", ignore_errors=True)

class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name, avatar="https://generated-images.perchance.org/image/777b73a7a72204c5417c33c77100317b96792fbfa4a55e7112a88053ba272cd9.jpeg"):
            st.write(f"**{message['name']}**: {message['content']}")
        return super()._process_received_message(message, sender, silent)

@st.cache_resource(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_gnn() -> Tuple[torch.nn.Module, MIMIC3Dataset, SampleEHRDataset]:
    dataset = MIMIC3Dataset(
        root='data/',
        tables=["DIAGNOSES_ICD","PROCEDURES_ICD","PRESCRIPTIONS","NOTEEVENTS_ICD"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 4}})},
    )

    mimic3sample = dataset.set_task(task_fn=medication_recommendation_mimic3_fn) # diagnosis prediction or 
                                                                                    # medication recommendation

    model = GNN(
        dataset=mimic3sample,
        feature_keys=["procedures", "symptoms", "diagnosis"],
        label_key="medications",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    return model, dataset, mimic3sample

@st.cache_data(hash_funcs={torch.Tensor: lambda _: None})
def med_recommended(y_prob: torch.Tensor, mimic_df: pd.DataFrame, top_k: int = 10) -> List[str]:
    atc = InnerMap.load("ATC")
    sorted_indices = []
    for i in range(len(y_prob)):
        top_indices = np.argsort(-y_prob[i, :])[:top_k]
        sorted_indices.append(top_indices)

    rec_medication = []

    # get the list of all medications in the dataset
    list_medications = mimic3sample.get_all_tokens('medications')

    sorted_indices = np.array(sorted_indices)
    # iterate over the top indexes for each sample in test_ds
    for (i, sample), top in zip(mimic_df.iterrows(), sorted_indices):
        # st.write(sorted_indices)

        # create an empty list to store the recommended medications for this sample
        sample_rec_medication = []

        # iterate over the top indexes for this sample
        for k in top:
            # append the medication at the i-th index to the recommended medications list for this sample
            sample_rec_medication.append(atc.lookup(list_medications[k]))

        # append the recommended medications for this sample to the recommended medications list
        rec_medication.append(sample_rec_medication)

    return rec_medication, sorted_indices

# @st.cache_resource(hash_funcs={GNN: lambda _: None, SampleEHRDataset: lambda _: None})
def explainability(model: GNN, explain_dataset: SampleEHRDataset, selected_idx: int, algorithm: str, task: str):
    explainer = HeteroGraphExplainer(
        algorithm=algorithm,
        dataset=explain_dataset,
        model=model,
        label_key=task,
        threshold_value=20,
        top_k=20,
        feat_size=128,
        root="./streamlit_results/",
    )

    visit_drug = explainer.subgraph['visit', 'medication'].edge_index
    # st.write(visit_drug)
    visit_drug = visit_drug.T

    n = 0
    for vis_drug in visit_drug:
        vis_drug = np.array(vis_drug)
        if vis_drug[1] == selected_idx:
            break
        n += 1

    # st.write(n)

    explainer.explain(n=n)
    explainer.explain_graph(k=0, human_readable=False, dashboard=True)
    explainer.explain_results(n=n)
    explainer.explain_results(n=n, doctor_type="Internist_Doctor")

    HtmlFile = open("streamlit_results/explain_graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=400, width=650)


# ---- SETTINGS PAGE ----
st.set_page_config(page_title="NameCare - Dashboard", page_icon="🩺", layout="wide")

# ---- SESSION STATE ----
if 'patient' not in st.session_state:
    st.session_state.patient = None
if 'name' not in st.session_state:
    st.session_state.name = None
if 'lastname' not in st.session_state:
    st.session_state.lastname = None
if 'gender_sign' not in st.session_state:
    st.session_state.gender_sign = None

# ---- Load model and dataset ----
model, dataset, mimic3sample = load_gnn()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)
model.eval()

# ---- DATAFRAME ----
mimic_df = pd.DataFrame(mimic3sample.samples)

# ---- SIDE BAR ----
st.sidebar.title('🩺 NameCare')
# st.sidebar.caption("TEXT")
st.sidebar.divider()

fake = Faker()

selected_patient = st.sidebar.selectbox(label='Select __patient__ n°: ', index=None, placeholder="Select a patient", options=mimic_df['patient_id'].unique())

if selected_patient is None:
    st.stop()

patient_dict = dataset.patients
patient_info = patient_dict[selected_patient]
gender = patient_info.gender

if selected_patient != st.session_state.patient:
    if gender == "M":
        first_name = fake.first_name_male()
        last_name = fake.last_name_male()
        gender_sign = "male_sign"
    elif gender == "F":
        first_name = fake.first_name_female()
        last_name = fake.last_name_female()
        gender_sign = "female_sign"
    else:
        first_name = "Name"
        last_name = "Unknown"

    st.session_state.patient = selected_patient
    st.session_state.name = ":red[" + first_name + "]"
    st.session_state.lastname = last_name
    st.session_state.gender_sign = gender_sign

patient = st.session_state.patient
name = st.session_state.name
lastname = st.session_state.lastname
gender_sign = st.session_state.gender_sign

mimic_df_patient = mimic_df[mimic_df['patient_id'] == selected_patient] # select all the rows with the selected patient
visit = st.sidebar.selectbox(label='Select __visit__ n°: ', options=mimic_df_patient['visit_id'].unique())

mimic_df_patient_visit = mimic_df_patient[mimic_df_patient['visit_id'] == visit] # select all the rows with the selected visit
task = st.sidebar.selectbox(label='Select __task__: ', options=['medications', 'diagnosis'])
algorithm = st.sidebar.selectbox(label='Select __Explainer algorithm__: ', options=['IG', 'GNNExplainer'])

# ---- MAIN PAGE ----
# st.header('🩺 E che munnezz!')
# st.dataframe(mimic_df)

# ---- Patient info ----
# st.subheader(":blue[DASHBOARD OF] ")
st.title("{} {} :{}:".format(name, lastname, gender_sign))
st.caption("Patient n°: {}  -  Gender: {}  -  Ethnicity: {}".format(patient, patient_info.gender, patient_info.ethnicity))

l1, r1 = st.columns([1.2, 2])
with l1:
    # dob = str(patient_info.birth_datetime)
    # dob = datetime.strptime(dob, "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y")
    # st.metric(label="📆 Date of birth", value=dob, delta="")
    # ---- Output model ----
    output = model(mimic_df_patient_visit['patient_id'],
                mimic_df_patient_visit['visit_id'],
                mimic_df_patient_visit['diagnosis'],
                mimic_df_patient_visit['procedures'],
                mimic_df_patient_visit['symptoms'],
                mimic_df_patient_visit['medications'])

    st.subheader("Recommended _medications_: ")
    med_recommended, med_indices = med_recommended(output['y_prob'], mimic_df_patient_visit)
    med_recommended = [[idx, item] for idx, item in zip(*med_indices, *med_recommended) if item]
    st.dataframe(med_recommended, column_config={"0": "ID", "1": "Recommended Medications"})

with r1:
    st.subheader("Explainability Graph? ")
    # dod = str(patient_info.death_datetime)
    # if dod != "None":
    #     dod = datetime.strptime(dod, "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y")
    #     st.metric(label="☠️ Date of death", value=dod, delta="")
    options = [item[1] for item in med_recommended if item]
    selected_medication = st.selectbox('Select the medication for explain', index=None, 
                                placeholder="Choice a medication to explain", options=options)

    if selected_medication is None:
        st.stop()

    st.write(f'You selected: __{selected_medication}__')
    selected_idx = [item[0] for item in med_recommended if item[1] == selected_medication]

    explain_sample = {}
    for visit_sample in mimic3sample.samples:
        if visit_sample['patient_id'] == patient and visit_sample['visit_id'] == visit:
            if visit_sample.get('drugs_hist') != None:
                del visit_sample['drugs_hist']
            explain_sample['test'] = visit_sample

    explain_dataset = SampleEHRDataset(list(explain_sample.values()), code_vocs="ATC")
    explainability(model, explain_dataset, selected_idx[0], algorithm, task)

####################### Scouter AI Component ##################################
st.header('👨‍⚕️ Doctor Explainer AI')
message = f"**TEXT**"
st.caption(message)

api_key = st.text_input("You need to enter the Open AI API Key:", placeholder="sk-...", type="password")
os.environ['OPENAI_API_KEY'] = api_key

if not(api_key):
    st.stop()

col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    with open("streamlit_results/medical_scenario.txt", "r") as f:
        medical_scenario = f.read()
    st.caption(medical_scenario)

with col2:
    with st.status("Recruiting doctor...", expanded=False) as status:
        # st.write("Searching for doctor...")
        # st.write("Found a doctor.")
        # st.write("Doctor recruited...")

        with open("streamlit_results/prompt_recruiter_doctors.txt", "r") as f:
            prompt_recruiter_doctors = f.read()

        client = OpenAIWrapper(api_key=os.environ['OPENAI_API_KEY'])
        response = client.create(messages=[{"role": "user", "content": prompt_recruiter_doctors}], 
                                temperature=0.3, 
                                seed=42, 
                                model="gpt-3.5-turbo")

        text = client.extract_text_or_completion_object(response)
        json_data = json.loads(text[0])
        with open("streamlit_results/recruited_doctors.json", "w") as f:
            json.dump(json_data, f, indent=4)

        for i, doctor in enumerate(json_data['doctors']):
            role = f"""{i+1} - {doctor['role'].replace(" ", "_")}"""
            st.write(role)

        status.update(label="Doctor recruited!", state="complete", expanded=True)

    st.button('Rerun')

    st.subheader("Analysis Proposition")

    with st.spinner("Doctors are thinking..."):
        with open("streamlit_results/prompt_internist_doctor.txt", "r") as f:
            prompt_internist_doctor = f.read()

        # OpenAI endpoint
        doctor = OpenAIWrapper(api_key=os.environ['OPENAI_API_KEY'])

        prompt_reunion = f"""###Instructions###\n"""
        prompt_reunion += f"""Based on YOUR analysis and the team's suggestions regarding the medication recommendation during the patient's visit: \n"""
        prompt_reunion += f"""- Discuss with your medical colleagues in the team, highlighting important aspects related to the patient's condition in relation to the administration of the medication and which have an impact on your decision of whether it is justifiable or unjustifiable.\n"""
        prompt_reunion += f"""- Elaborate on your initial opinion in the light of the discussions and opinions of the other doctors, giving more emphasis to the aspects you consider appropriate. \n\n"""

        prompt_reunion += f"""###Analysis###\n"""
        prompt_reunion += f"""Analysis Propositions: \n"""
        for i in range(len(json_data['doctors'])):
            with st.chat_message(name="user", avatar="https://generated-images.perchance.org/image/777b73a7a72204c5417c33c77100317b96792fbfa4a55e7112a88053ba272cd9.jpeg"):
                analysis = """"""
                analysis += f"""**Doctor**: {json_data['doctors'][i]['role'].replace(" ", "_")}\n\n"""
                response = doctor.create(messages=[
                                                    {"role": "system", 
                                                        "content": json_data['doctors'][i]['description']},
                                                    {"role": "user", 
                                                        "content": prompt_internist_doctor}
                                                    ], 
                                            temperature=0.5, 
                                            model="gpt-3.5-turbo") 
                analysis += "**Analysis**: " + doctor.extract_text_or_completion_object(response)[0]
                st.markdown(analysis)
                prompt_reunion += f"""{analysis}"""
                prompt_reunion += f"\n--------------------------------------------------\n\n"

        prompt_reunion += f"""###Desired Behavior###\n"""
        prompt_reunion +=f"""You MUST summarise your opinion on the recommendation in a text of 25-WORDS, which helps the INTERNIST DOCTOR to make a final decision."""

st.header('💬 Collaborative Discussion')

with st.spinner("Doctors are discussing..."):
    config_list = [
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ['OPENAI_API_KEY']
        }
    ]

    llm_config={
        "timeout": 500,
        "seed": 42,
        "config_list": config_list,
        "temperature": 0.5
    }

    doc = []

    for i in range(len(json_data['doctors'])):
        doc.append(AssistantAgent(
            name=json_data['doctors'][i]['role'].replace(" ", "_"), # Nome del dottore esperto
            llm_config=llm_config,
            system_message="You are the " + json_data['doctors'][i]['role'].replace(" ", "_") + "."
        ))

    internist_sys_message = f"""You are the INTERNIST DOCTOR who deals with the overall assessment of the patient and his pathology, integrating the different therapies, so that you can manage his health in a comprehensive manner. \n"""
    internist_sys_message += f"""DIRECTIVE: ONLY AFTER the doctors have provided the most relevant aspects on the recommendation of the medication to be administered to the patient, you MUST explain your opinion on the recommendation based on the opinions of the doctors and your own medical knowledge, taking into account the potential benefits or risks. \n"""
    internist_sys_message += f"""EXPLAIN your considerations and ONLY THEN reply with the FINAL DECISION: JUSTIFIABLE/NON JUSTIFIABLE to close the conversation."""

    doc.append(TrackableUserProxyAgent(
        name="internist_doctor", # Recruiter che passa il Report della visita del paziente
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(("TERMINATE")),
        code_execution_config=False,
        llm_config=llm_config,
        system_message=internist_sys_message
    ))

    groupchat = autogen.GroupChat(agents=doc, 
                                messages=[], 
                                max_round=(len(doc)+1), 
                                speaker_selection_method="round_robin")

    manager = autogen.GroupChatManager(groupchat=groupchat, 
                                    llm_config=llm_config, 
                                    max_consecutive_auto_reply=1)

    doc[-1].initiate_chat(
        manager,
        message=prompt_reunion,
    )

    with st.chat_message(name="user", avatar="https://generated-images.perchance.org/image/777b73a7a72204c5417c33c77100317b96792fbfa4a55e7112a88053ba272cd9.jpeg"):
        internist = list(manager.chat_messages.values())
        st.write(f"**{internist[0][6]['name']}**: {internist[0][6]['content']}")