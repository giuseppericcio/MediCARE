from typing import Tuple, List
from faker import Faker
# from datetime import datetime

import os
import json
import torch
import shutil
import autogen
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from autogen import OpenAIWrapper, AssistantAgent, UserProxyAgent

from pyhealth.medcode import InnerMap
from pyhealth.datasets import MIMIC3Dataset, SampleEHRDataset
from pyhealth.tasks import medication_recommendation_mimic3_fn, diagnosis_prediction_mimic3_fn
from pyhealth.models import GNN
from pyhealth.explainer import HeteroGraphExplainer

PATH_MED = "model/medication_recommendation/best.ckpt"
PATH_DIAG = "model/diagnosis_prediction/best.ckpt"
# shutil.rmtree(".cache/", ignore_errors=True)

class TrackableUserProxyAgent(UserProxyAgent):
    t = 0
    def _process_received_message(self, message, sender, silent):
        global t  # Declare t as a global variable
        with st.chat_message(sender.name, avatar="streamlit_images/{}.png".format(self.t)):
            st.write(f"**{message['name'].replace('_',' ')}**: {message['content']}")
            self.t += 1
            if self.t == 4:
                self.t = 0
        st.divider()
        return super()._process_received_message(message, sender, silent)

@st.cache_resource(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_gnn() -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module,
                        MIMIC3Dataset, SampleEHRDataset, SampleEHRDataset]:
    dataset = MIMIC3Dataset(
        root='data/',
        tables=["DIAGNOSES_ICD","PROCEDURES_ICD","PRESCRIPTIONS","NOTEEVENTS_ICD"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 4}})},
    )

    mimic3sample_med = dataset.set_task(task_fn=medication_recommendation_mimic3_fn)
    mimic3sample_diag = dataset.set_task(task_fn=diagnosis_prediction_mimic3_fn)

    model_med_ig = GNN(
        dataset=mimic3sample_med,
        convlayer="GraphConv",
        feature_keys=["procedures", "symptoms", "diagnosis"],
        label_key="medications",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    model_med_gnn = GNN(
        dataset=mimic3sample_med,
        convlayer="GraphConv",
        feature_keys=["procedures", "symptoms", "diagnosis"],
        label_key="medications",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    model_diag_ig = GNN(
        dataset=mimic3sample_diag,
        convlayer="GraphConv",
        feature_keys=["procedures", "symptoms", "medications"],
        label_key="diagnosis",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    model_diag_gnn = GNN(
        dataset=mimic3sample_diag,
        convlayer="GraphConv",
        feature_keys=["procedures", "symptoms", "medications"],
        label_key="diagnosis",
        k=0,
        embedding_dim=128,
        hidden_channels=128
    )

    return model_med_ig, model_med_gnn, model_diag_ig, model_diag_gnn, dataset, mimic3sample_med, mimic3sample_diag

@st.cache_data(hash_funcs={torch.Tensor: lambda _: None})
def get_list_ouput(y_prob: torch.Tensor, mimic_df: pd.DataFrame, task: str, top_k: int = 10) -> List[str]:
    sorted_indices = []
    for i in range(len(y_prob)):
        top_indices = np.argsort(-y_prob[i, :])[:top_k]
        sorted_indices.append(top_indices)

    list_output = []

    # get the list of all labels in the dataset
    if task == "medications":
        list_labels = mimic3sample.get_all_tokens('medications')
        atc = InnerMap.load("ATC")
    elif task == "diagnosis":
        list_labels = mimic3sample.get_all_tokens('diagnosis')
        icd9 = InnerMap.load("ICD9CM")

    sorted_indices = list(sorted_indices)
    # iterate over the top indexes for each sample in test_ds
    for (i, sample), top in zip(mimic_df.iterrows(), sorted_indices):
        # st.write(sorted_indices)

        # create an empty list to store the recommended medications for this sample
        sample_list_output = []

        # iterate over the top indexes for this sample
        for k in top:
            # append the medication at the i-th index to the recommended medications list for this sample
            if task == "medications":
                sample_list_output.append(atc.lookup(list_labels[k]))
            elif task == "diagnosis":
                if list_labels[k].startswith("E"):
                    list_labels[k] = list_labels[k] + "0"
                sample_list_output.append(icd9.lookup(list_labels[k]))

        # append the recommended medications for this sample to the recommended medications list
        list_output.append(sample_list_output)

    return list_output, sorted_indices

# @st.cache_resource(hash_funcs={GNN: lambda _: None, SampleEHRDataset: lambda _: None})
def explainability(model: GNN, explain_dataset: SampleEHRDataset, selected_idx: int, 
                   visualization: str, algorithm: str, task: str, threshold: int):
    explainer = HeteroGraphExplainer(
        algorithm=algorithm,
        dataset=explain_dataset,
        model=model,
        label_key=task,
        threshold_value=threshold,
        top_k=threshold,
        feat_size=128,
        root="./streamlit_results/",
    )

    if task == "medications":
        visit_drug = explainer.subgraph['visit', 'medication'].edge_index
        visit_drug = visit_drug.T

        n = 0
        for vis_drug in visit_drug:
            vis_drug = np.array(vis_drug)
            if vis_drug[1] == selected_idx:
                break
            n += 1
    elif task == "diagnosis":
        visit_diag = explainer.subgraph['visit', 'diagnosis'].edge_index
        visit_diag = visit_diag.T

        n = 0
        for vis_diag in visit_diag:
            vis_diag = np.array(vis_diag)
            if vis_diag[1] == selected_idx:
                break
            n += 1

    # st.write(n)

    explainer.explain(n=n)
    if visualization == "Explainable":
        explainer.explain_graph(k=0, human_readable=True, dashboard=True)
    else:
        explainer.explain_graph(k=0, human_readable=False, dashboard=True)

    explainer.explain_results(n=n)
    explainer.explain_results(n=n, doctor_type="Internist_Doctor")

    HtmlFile = open("streamlit_results/explain_graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=520)


# ---- SETTINGS PAGE ----
st.set_page_config(page_title="MHGCORECare - Dashboard", page_icon="🩺", layout="wide")

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# ---- SESSION STATE ----
if 'patient' not in st.session_state:
    st.session_state.patient = None
if 'name' not in st.session_state:
    st.session_state.name = None
if 'lastname' not in st.session_state:
    st.session_state.lastname = None
if 'gender_sign' not in st.session_state:
    st.session_state.gender_sign = None


# ---- LOAD MODEL AND DATASET ----
# ---- SIDE BAR ----
st.sidebar.image(".\streamlit_images\logo_icon.png")
# st.sidebar.title('MedHGCORECare')
# st.sidebar.caption("TEXT")
st.sidebar.divider()

# ---- MAIN PAGE ----
st.title(":rainbow[GREATCARE] :grey[(Alpha-Test)]")
st.markdown("Welcome to the dashboard of the GREATCARE project!")

desc = st.empty()
desc1 = st.empty()
desc.caption("The dashboard is divided into two main sections: the **Graphs Enriched with External Knowledge and Clinical Text for Personalised Predictive Healthcare (GREAT) module** and the **COllaborative Reasoning Engine (CORE) module**. The **GREAT module** is responsible for processing the patient's medical history and generating recommendations for the doctor. The **Medical Collaborative Agents REasoning over Interpretable Heterogeneous Graphs module (CARE)** is responsible for generating the analysis of the doctors' proposals and the collaborative discussion between the medical team members for the final decision on the patient's treatment.")
desc1.caption("**⏳ WAIT MINUTES FOR THE LOADING OF THE MODELS AND THE DATASET**")

model_med_ig, model_med_gnn, model_diag_ig, model_diag_gnn, \
    dataset, mimic3sample_med, mimic3sample_diag = load_gnn()
checkpoint_MED = torch.load(PATH_MED)
checkpoint_DIAG = torch.load(PATH_DIAG)

desc.empty()
desc1.empty()

fake = Faker()

selected_patient = None
if selected_patient is None:
    
    placeholder2 = st.empty()
    with placeholder2.expander("⚠️ **Before using the framework, read the disclaimer for the use of Framework**"):
        disclaimer = f"""

        The use of our Healthcare framework based on MIMIC III (https://physionet.org/content/mimiciii/1.4/) is subject to the following terms and warnings:

        **Research and Decision Support Purpose:** Our framework has been developed primarily for research and decision support in the healthcare context. The information and recommendations generated should not replace the professional judgment of qualified healthcare practitioners but may be utilized as support for the final decision by the doctor or the directly involved party.

        **Data Origin:** The processed healthcare data originates from the MIMIC III database and undergoes enrichment and modeling through the application of Heterogeneous Graph Neural Network. It is important to note that the original data may contain variations and limitations, and the accuracy of the processed information depends on the quality of the input data.

        **Medical Recommendations:** The drug and diagnosis recommendations generated by the framework are hypothetical and based on Graph Neural Network learning models. These should not be considered definitive prescriptions, and the final decision regarding patient treatment should be made by a qualified medical professional.

        **Human Readable Explanations:** The embedded explainability system in the framework utilizes graph explainability models and Large Language Models (LLM) to generate understandable explanations for end-users, such as physicians. However, these explanations are interpretations of the model results and may not fully reflect the complexity of medical reasoning.

        **Framework Limitations:** Our framework has intrinsic limitations, including those related to the quality of input data, the characteristics of the machine learning model, and the dynamics of the healthcare context. Users are encouraged to exercise caution in interpreting the provided information.

        **User Responsibility:** Users accessing and utilizing our framework are responsible for the accurate interpretation of the provided information and for making appropriate decisions based on their clinical judgment. The creators assume no responsibility for any consequences arising from improper use or misinterpretation of the information generated by the framework.

        By using our healthcare data processing framework, the user agrees to comply with these conditions. The continuous evolution of the fields of medicine and technology may necessitate periodic updates to this disclaimer.
        """

        st.subheader("Disclaimer")
        st.info(disclaimer)
        agree = st.checkbox("I accept and have read the disclaimer!")
        placeholder1 = st.empty()
        placeholder1.warning("You must accept the disclaimer to use the framework!", icon="⚠️")

        if not(agree):
            st.stop()

        placeholder1.empty()
        placeholder2.info("You can now use the framework! 🎉 Please select the task and select a patient! 🩺")
        task = st.sidebar.selectbox(label='Select __task__: ', index=None, placeholder="Select type of task", options=['medications', 'diagnosis'])

        if task is None:
            st.stop()
        elif task == "medications":
            mimic3sample = mimic3sample_med
        elif task == "diagnosis":
            mimic3sample = mimic3sample_diag

        mimic_df = pd.DataFrame(mimic3sample.samples)

        selected_patient = st.sidebar.selectbox(label='Select __patient__ n°: ', index=None, placeholder="Select a patient", options=mimic_df['patient_id'].unique())
        while selected_patient is None:
            st.stop()

placeholder2.empty()

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
    st.session_state.name = ":blue[" + first_name + "]"
    st.session_state.lastname = last_name
    st.session_state.gender_sign = gender_sign

patient = st.session_state.patient
name = st.session_state.name
lastname = st.session_state.lastname
gender_sign = st.session_state.gender_sign

mimic_df_patient = mimic_df[mimic_df['patient_id'] == selected_patient] # select all the rows with the selected patient

for i in range(len(mimic_df_patient)):
    if i == len(mimic_df_patient) - 1:
        last_visit = mimic_df_patient.iloc[[i]]

# visit = st.sidebar.selectbox(label='Select __visit__ n°: ', options=mimic_df_patient['visit_id'].unique())
# task = st.sidebar.selectbox(label='Select __task__: ', options=['medications', 'diagnosis'])
# algorithm = st.sidebar.selectbox(label='Select __Explainer algorithm__: ', options=['IG', 'GNNExplainer'])


# ---- MAIN PAGE ----
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

    st.subheader("📋 Patient's complete medical history")
    st.caption("The following table shows the *complete* medical history of the patient n°: **{}**.".format(patient))

    visit = st.selectbox(label='🏥 Select __visit__ n°: ', options=mimic_df_patient['visit_id'].unique())
    if visit:
        mimic_df_patient_visit = mimic_df_patient[mimic_df_patient['visit_id'] == visit] # select all the rows with the selected visit
        if task == "medications":
            mimic_df_patient_visit_filtered = mimic_df_patient_visit.drop(columns=['visit_id', 'patient_id', 'drugs_hist'])
        elif task == "diagnosis":
            mimic_df_patient_visit_filtered = mimic_df_patient_visit.drop(columns=['visit_id', 'patient_id'])

        atc = InnerMap.load("ATC")
        icd9 = InnerMap.load("ICD9CM")
        icd9_proc = InnerMap.load("ICD9PROC")

        for column in mimic_df_patient_visit_filtered.columns:
            with st.expander("{}".format(column)):
                try:
                    if column == "medications":
                        if task == "medications":
                            med_history = [[med, atc.lookup(med)] for med in mimic_df_patient_visit_filtered[column].explode() if med]
                        elif task == "diagnosis":
                            med_history = [[med, atc.lookup(med)] for med in (mimic_df_patient_visit_filtered[column].explode()).explode() if med]
                        st.dataframe(med_history, hide_index=True, column_config={"0": "ATC", "1": "Description"})
                    elif column == "diagnosis":
                        if task == "medications":
                            col_history = [[idx, icd9.lookup(idx)] for idx in (mimic_df_patient_visit_filtered[column].explode()).explode() if idx]
                        elif task == "diagnosis":
                            col_history = [[idx+'0', icd9.lookup(idx+'0')] if idx.startswith('E') else [idx, icd9.lookup(idx)] for idx in mimic_df_patient_visit_filtered[column].explode() if idx]
                        st.dataframe(col_history, hide_index=True, column_config={"0": "ICD9", "1": "Description"})
                    elif column == "symptoms":
                        col_history = [[idx, icd9.lookup(idx)] for idx in (mimic_df_patient_visit_filtered[column].explode()).explode() if idx]
                        st.dataframe(col_history, hide_index=True, column_config={"0": "ICD9", "1": "Description"})
                    elif column == "procedures":
                        col_history = [[idx, icd9_proc.lookup(idx)] for idx in (mimic_df_patient_visit_filtered[column].explode()).explode() if idx]
                        st.dataframe(col_history, hide_index=True, column_config={"0": "ICD9", "1": "Description"})
                except:
                    st.write("No data available for this column.")

    if task == "medications":
        st.subheader("🧾 Recommended _medications_ of the Last Visit")
        st.caption(f"""The following medications are recommended for the patient during the last visit n°: {format(last_visit['visit_id'].item())}. \nThe recommendations are based on the output probabilities generated by the **GNN (_Graph Neural Network_)** model.""")

        model_med_ig.load_state_dict(checkpoint_MED)
        model_med_gnn.load_state_dict(checkpoint_MED)
        model = model_med_ig
    elif task == "diagnosis":
        st.subheader("🧾 Recommended Predicted _diagnosis_ of the Last Visit")
        st.caption(f"""The following diagnosis are predicted for the patient during the last visit n°: {format(last_visit['visit_id'].item())}. \nThe predictions are based on the output probabilities generated by the **GNN (_Graph Neural Network_)** model.""")

        model_diag_ig.load_state_dict(checkpoint_DIAG)
        model_diag_gnn.load_state_dict(checkpoint_DIAG)
        model = model_diag_ig

    model.eval()
    # ---- Output model ----
    output = model(last_visit['patient_id'],
                last_visit['visit_id'],
                last_visit['diagnosis'],
                last_visit['procedures'],
                last_visit['symptoms'],
                last_visit['medications'])

    list_output, list_indices = get_list_ouput(output['y_prob'], last_visit, task)
    list_output = [[idx, item] for idx, item in zip(*list_indices, *list_output) if item]
    if task == "medications":
        st.dataframe(list_output, column_config={"0": "ID", "1": "Recommended Medications"}, height=None, width=None)
    elif task == "diagnosis":
        st.dataframe(list_output, column_config={"0": "ID", "1": "Predicted Diagnosis"}, height=None, width=None)

with r1:
    st.subheader(f"""🗣 *Why* the model recommends these {task}?""")

    r1l1, r1c1, r1r1 = st.columns(3)
    with r1l1:
        visualization = st.radio("Visualization", options=["Explainable", "Interpretable"], horizontal=True)
    with r1c1:
        algorithm = st.radio("Algorithm", options=["IG", "GNNExplainer"], horizontal=True)
    with r1r1:
        threshold = st.slider("Threshold", min_value=10, max_value=50, value=15, step=5, format=None, key=None)

    if task == "medications" and algorithm == "IG":
        model = model_med_ig
    elif task == "medications" and algorithm == "GNNExplainer":
        model = model_med_gnn
    elif task == "diagnosis" and algorithm == "IG":
        model = model_diag_ig
    elif task == "diagnosis" and algorithm == "GNNExplainer":
        model = model_diag_gnn
    model.eval()

    if task == "medications":
        st.caption(f"""The following graph shows the explainability of the model's decision on the recommended medications for the patient during the visit n°: {format(visit)}. \nThe explainability is based on the **{algorithm} (_{task}_)** algorithm.""")
        options = [item[1] for item in list_output if item]
        selected_label = st.selectbox('Select the medication for explain', index=None, 
                                    placeholder="Choice a medication from Recommended medications ranking to explain", options=options)
    elif task == "diagnosis":
        st.caption(f"""The following graph shows the explainability of the model's decision on the predicted diagnosis for the patient during the visit n°: {format(visit)}. \nThe explainability is based on the **{algorithm} (_{task}_)** algorithm.""")
        options = [item[1] for item in list_output if item]
        selected_label = st.selectbox('Select the diagnosis for explain', index=None, 
                                    placeholder="Choice a diagnosis from Predicted diagnosis ranking to explain", options=options)

    if selected_label is None:
        st.stop()

    #st.write(f'You selected: __{selected_label}__')
    selected_idx = [item[0] for item in list_output if item[1] == selected_label]

    st.caption("Legend of the graph:")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3])

    with col1:
        st.markdown(
            """
            <style>
            #square1 {
                width: 20px;
                height: 20px;
                background: #20b2aa;
                border-radius: 3px;
            }
            </style>
            <div id="square1"></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
            #square2 {
                width: 20px;
                height: 20px;
                background: #fa8072;
                border-radius: 3px;
                margin-top: 20px;

            }
            </style>
            <div id="square2"></div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.caption("Patient")

        st.caption("Visit")

    with col3:
        st.markdown(
            """
            <style>
            #square3 {
                width: 20px;
                height: 20px;
                background: #cd853f;
                border-radius: 3px;
            }
            </style>
            <div id="square3"></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <style>
            #square4 {
                width: 20px;
                height: 20px;
                background: #da70d6;
                border-radius: 3px;
                margin-top: 20px;
            }
            </style>
            <div id="square4"></div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.caption("Diagnosis")

        st.caption("Procedures")
        

    with col5:
        st.markdown(
            """
            <style>
            #square5 {
                width: 20px;
                height: 20px;
                background: #98fb98;
                border-radius: 3px;
            }
            </style>
            <div id="square5"></div>
            """,
            unsafe_allow_html=True,
        )
        
    with col6:
        st.caption("Symptoms")
        

    with col7:
        st.markdown(
            """
            <style>
            #square6 {
                width: 20px;
                height: 20px;
                background: #87ceeb;
                border-radius: 3px;
            
            }
            </style>
            <div id="square6"></div>
            """,
            unsafe_allow_html=True,
        )
    
    with col8:
        st.caption("Medications")
    
    explain_sample = {}
    for visit_sample in mimic3sample.samples:
        if visit_sample['patient_id'] == patient and visit_sample['visit_id'] == last_visit['visit_id'].item():
            if visit_sample.get('drugs_hist') != None:
                del visit_sample['drugs_hist']
            explain_sample['test'] = visit_sample

    explain_dataset = SampleEHRDataset(list(explain_sample.values()), code_vocs="ATC")
    explainability(model, explain_dataset, selected_idx[0], visualization, algorithm, task, threshold)


####################### CARE AI module ##################################
st.header('🩺🧠 *C*ollaborative *A*gents *RE*asoning')
st.caption("The following section is dedicated to the CARE module, which is responsible for generating the analysis of the doctors' proposals and the collaborative discussion between the medical team members for the final decision on the patient's treatment.")

api_key = st.text_input("You need to enter the Open AI API Key:", placeholder="sk-...", type="password")
os.environ['OPENAI_API_KEY'] = api_key

if not(api_key):
    st.stop()

col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    with open("streamlit_results/medical_scenario.txt", "r") as f:
        medical_scenario = f.read()
    st.subheader("📄 Important Medical Scenario")
    st.caption(f"The following scenario of the patient in the visit n°: {format(visit)} is provided by the medical team.")
    st.markdown(medical_scenario)

with col2:
    st.subheader("👨‍⚕️🔎 Doctor Recruiter")
    st.caption("The doctor recruiter is responsible for recruiting the medical team to help the internist doctor make a final decision on the patient's during the collaborative discussion.")
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
            role = f"""**🥼 {doctor['role'].replace("_", " ")}**"""
            st.markdown(role)
            st.write(doctor['description'])
            if i != len(json_data['doctors'])-1:
                st.divider()

        status.update(label="Doctor recruited!", state="complete", expanded=True)

    st.button('Rerun')

    st.subheader("Analysis Proposition")

    with st.spinner("Doctors are thinking...") as status_docs:
        with open("streamlit_results/prompt_internist_doctor.txt", "r") as f:
            prompt_internist_doctor = f.read()

        # OpenAI endpoint
        doctor = OpenAIWrapper(api_key=os.environ['OPENAI_API_KEY'])

        if task == "medications":
            prompt_reunion = f"""Based on your assessment and the medical team's recommendations regarding medication administration during the patient visit:\n"""
            prompt_reunion += f"""Confront with your medical colleagues, highlighting relevant aspects related to the patient's condition and the administration of the drug. Underline the crucial elements that influence your decision on its justification or unjustification in 30 words.\n"""
            prompt_reunion += f"""\nAnalysis of doctors' proposals\n\n"""
        elif task == "diagnosis":
            prompt_reunion = f"""Based on your assessment and the medical team's recommendations regarding diagnosis during the patient visit:\n"""
            prompt_reunion += f"""Confront with your medical colleagues, highlighting relevant aspects related to the patient's condition and the diagnosis. Underline the crucial elements that influence your decision on its justification or unjustification in 30 words.\n"""
            prompt_reunion += f"""\nAnalysis of doctors' proposals\n\n"""


        for i in range(len(json_data['doctors'])):
            with st.status(f"The 👨‍⚕️ {json_data['doctors'][i]['role'].replace('_', ' ')} is analysing ...", expanded=False) as status_doc:
                with st.chat_message(name="user", avatar="streamlit_images/{}.png".format(i)):
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
                    st.markdown(f"**Analysis**: {doctor.extract_text_or_completion_object(response)[0]}")
                    status_doc.update(label="The 👨‍⚕️ {} analysed!".format(json_data['doctors'][i]['role'].replace('_', ' ')), state="complete", expanded=True)
                    prompt_reunion += f"""{analysis}"""
                    prompt_reunion += f"\n--------------------------------------------------\n\n"

image, text = st.columns([0.5, 2])

with image: 
    st.image("streamlit_images/collaborative.png")

with text:
    st.subheader('*CARE* Discussion')

st.caption("The following discussion is based on the **Large Language Model** (LLM) **GPT-3.5-turbo**. The LLM is responsible for generating the discussion between the medical team members for the final decision on the patient's treatment.")

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
            system_message="As a " + json_data['doctors'][i]['role'].replace(" ", "_") + ". Discuss with other medical experts in the team to help the INTERNIST DOCTOR make a final decision. Avoid postponing further examinations and repeating opinions given in the analysis, but explain in a logical and concise manner why you are making this final decision."))
            

    if task == "medications":
        # internist_sys_message = f"""As an INTERNIST DOCTOR, you have the task of globally evaluating and managing the patient's health and pathology.\n"""
        # internist_sys_message += f"""ONLY AFTER listening to medical specialists' opinions on medication recommendations, provide your assessment based on your medical expertise. Explore the possible benefits and risks of the decision.\n"""
        # internist_sys_message += f"""EXPLAIN your considerations and, SUBSEQUENTLY, determine a FINAL DECISION taking into account the majority of opinions: conclude the discussion with "JUSTIFIABLE" or "UNJUSTIFIABLE"."""
    
        internist_sys_message = f"""As an INTERNIST DOCTOR, you have the task of globally evaluating and managing the patient's health and pathology.\n"""
        internist_sys_message += f"""In the light of the entire discussion, you must provide a final schematic report to the doctor based on the recommendation and the doctors' opinions."""

    elif task == "diagnosis":
        internist_sys_message = f"""As an INTERNIST DOCTOR, you have the task of globally evaluating and managing the patient's health and pathology.\n"""
        internist_sys_message += f"""ONLY AFTER listening to medical specialists' opinions on diagnosis predictions, provide your assessment based on your medical expertise. Explore the possible benefits and risks of the decision.\n"""
        internist_sys_message += f"""EXPLAIN your considerations and, SUBSEQUENTLY, determine a FINAL DECISION taking into account the majority of opinions: conclude the discussion with "JUSTIFIABLE" or "UNJUSTIFIABLE"."""

    doc.append(TrackableUserProxyAgent(
        name="internist_doctor", # Recruiter che passa il Report della visita del paziente
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(("JUSTIFIABLE", "UNJUSTIFIABLE")),
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

    with st.chat_message(name="user", avatar="streamlit_images/internist.png"):
        internist = list(manager.chat_messages.values())
        st.write(f"**{internist[0][6]['name'].replace('_',' ')}**: {internist[0][6]['content']}")
